# Author Toshihiko Aoki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training and evaluate helper."""

import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from .utils import set_seeds, save, load, get_device
from .optimization import get_step


class Helper(object):

    def __init__(self, seeds=20151106, device=None, fp16=False, cli_interval=10):
        super().__init__()
        set_seeds(seeds)
        self.num_gpu = 0

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(get_device())

        if torch.cuda.is_available() and self.device is not 'cpu':
            torch.backends.cudnn.benchmark = True
            self.num_gpu = torch.cuda.device_count()
            self.fp16 = fp16 & torch.cuda.is_available()
        else:
            self.fp16 = False

        print("device: {} num_gpu: {} fp16: {}".format(self.device, self.num_gpu, self.fp16))
        self.cli_interval = cli_interval

    def load_model(self, model, model_path, optimizer=None, strict=True):
        load(model, model_path, self.device, optimizer, strict)

    def train(
        self,
        process,
        model,
        dataset,
        sampler,
        optimizer,
        scheduler,
        batch_size=1,
        epochs=20,
        model_file=None,
        save_dir='train/',
        per_save_epochs=-1,
        adjustment_every_epoch=None,
        adjustment_every_step=None,
        opt_level='O2',
        max_grad_norm=1.0,
        cpu_param_optimizer=None
    ):
        model.to(self.device)
        model.train()
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)
        if model_file is not None and model_file is not '':
            # optimizer attributes override
            if self.num_gpu > 1 and hasattr(model, 'module'):
                load(model.module, model_file, self.device, optimizer)
            else:
                load(model, model_file, self.device, optimizer)
        global_step = get_step(optimizer)
        print('Optimizer start steps : {:d}'.format(global_step))
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        for e in range(epochs):
            total_loss = window_loss = window_best = .0
            total_steps = 0
            iter_bar = tqdm(
                data_loader, desc="E-{:0=2} : XX.XXXX avg loss ".format(e), position=0, mininterval=self.cli_interval)
            for step, batch in enumerate(iter_bar):
                optimizer.zero_grad()

                batch = tuple(t.to(self.device) for t in batch)
                loss = process(batch, model, iter_bar, e, step)

                if self.num_gpu > 1:
                    loss = loss.mean()
                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                total_steps += 1
                total_loss += loss.item()
                iter_bar.set_description("E-{:0=2} : {:2.4f} avg loss ".format(e, total_loss / total_steps),
                                         refresh=False)
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if cpu_param_optimizer is not None:
                    for (_, param_optimizer), (_, param_model) in zip(cpu_param_optimizer, model.named_parameters()):
                        if param_model.grad is not None:
                            if param_optimizer.grad is None:
                                param_optimizer.grad = torch.nn.Parameter(
                                    param_optimizer.data.new().resize_(*param_optimizer.data.size()))
                            param_optimizer.grad.data.copy_(param_model.grad.data)
                        else:
                            param_optimizer.grad = None
                    optimizer.step()
                    for (_, param_optimizer), (_, param_model) in zip(cpu_param_optimizer, model.named_parameters()):
                        param_model.data.copy_(param_optimizer.data)

                else:
                    optimizer.step()
                scheduler.step()
                global_step += 1

                if adjustment_every_step is not None:
                    window_loss, window_best = adjustment_every_step(
                        model, dataset, loss.item(), total_steps, global_step,
                        optimizer, batch_size, window_loss, window_best)

            if per_save_epochs > 0 and (e + 1) % per_save_epochs is 0:
                output_model_file = os.path.join(save_dir, "train_model.pt")
                save(model, output_model_file, optimizer)

            if adjustment_every_epoch is not None:
                adjustment_every_epoch(model, dataset, total_loss, total_steps, optimizer)

        return total_loss / total_steps

    def evaluate(
        self,
        process,
        model,
        dataset,
        sampler,
        batch_size,
        model_file=None,
        examples_reports=None,
        compute_epoch_score=None,
        adjustment_every_epoch=None,
        epochs=1,
        evaluate_score=None,
    ):
        model.to(self.device)
        model.eval()
        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)
        if model_file is not None:
            load(model, model_file, self.device)
            print('loaded ; ' + str(model_file))

        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        scores = []
        for e in range(epochs):
            global_step = 0
            total_loss = 0.0
            total_steps = 0
            examples = []
            iter_bar = tqdm(
                data_loader, desc="E-{:0=2} : XX.XXXX avg loss ".format(e), position=0, mininterval=self.cli_interval)
            for step, batch in enumerate(iter_bar):
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    loss, example = process(batch, model, iter_bar, step)
                examples.append(example)
                if self.num_gpu > 1:
                    loss = loss.mean()

                total_loss += loss.item()
                total_steps += 1
                iter_bar.set_description("E-{:0=2} : {:2.4f} avg loss ".format(e, total_loss / total_steps),
                                         refresh=False)
                global_step += 1

            if examples_reports is not None:
                examples_reports(examples)

            if compute_epoch_score is not None:
                epoch_score = compute_epoch_score(examples)
            else:
                epoch_score = total_loss / total_steps  # default loss

            scores.append(epoch_score)

            if adjustment_every_epoch is not None:
                adjustment_every_epoch(model, dataset, total_loss, total_steps)

        if evaluate_score is not None:
            score = evaluate_score(scores)
        else:
            from statistics import mean
            score = mean(scores)  # default mean loss

        return score

    def predict(
        self,
        process,
        model,
        dataset,
        sampler=None,
        batch_size=1,
        model_file=None
    ):
        model.to(self.device)
        model.eval()

        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)
        if model_file is not None:
            load(model, model_file, self.device)
        model.eval()

        if sampler is None:
            sampler = SequentialSampler(dataset)

        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        predicts = []
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                predict = process(batch, model, step)
                predicts.append(predict)

            cpu_device = torch.device('cpu')
            tuple(t.to(cpu_device) for t in batch)
            torch.cuda.empty_cache()

        return predicts

