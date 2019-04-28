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
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from . utils import set_seeds, save, load, get_device


class Helper(object):

    def __init__(self, seeds=20151106, device=None, fp16=False):
        super().__init__()
        set_seeds(seeds)
        self.num_gpu = 0

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(get_device())

        if torch.cuda.is_available() and self.device is not 'cpu':
            self.num_gpu = torch.cuda.device_count()

        self.fp16 = fp16 & torch.cuda.is_available()

        print("device: {} num_gpu: {} fp16: {}".format(self.device, self.num_gpu, self.fp16))

    def load_model(self, model, model_path, optimizer=None):
        load(model, model_path, self.device, optimizer)

    def train(
        self,
        process,
        model,
        dataset,
        sampler,
        optimizer,
        batch_size=1,
        epoch=20,
        model_file=None,
        save_dir='train/',
        per_save_epoc=-1,
        adjustment_every_epoch=None,
        adjustment_every_step=None
    ):

        if self.fp16:
            model.half()
        model.to(self.device)
        if self.num_gpu > 1:  # not test
            model = torch.nn.DataParallel(model)
        if model_file is not None and model_file is not '':
            # warmup_steps over-ride
            load(model, model_file, self.device, optimizer)
        model.train()
        global_step = optimizer.get_step()
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        for e in range(epoch):
            total_loss = 0.0
            total_steps = 0
            iter_bar = tqdm(
                dataloader, desc="E-{:0=2} : XX.XXXX avg loss ".format(e), position=0)
            for step, batch in enumerate(iter_bar):
                batch = tuple(t.to(self.device) for t in batch)

                loss = process(batch, model, iter_bar, e, step)

                if self.num_gpu > 1:
                    loss = loss.mean()
                if self.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if adjustment_every_step is not None:
                    adjustment_every_step(model, dataset, loss, global_step, optimizer)

                total_loss += loss.item()
                total_steps += 1
                iter_bar.set_description("E-{:0=2} : {:2.4f} avg loss ".format(e, total_loss / total_steps))
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if per_save_epoc > 0 and (e + 1) % per_save_epoc is 0:
                output_model_file = os.path.join(save_dir, "train_model_" + str(e) + "_" + str(global_step) + ".pt")
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
        epoch=1,
        evaluate_score=None,
    ):
        if self.fp16 and torch.cuda.is_available():
            model.half()
        model.to(self.device)
        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)

        if model_file is not None:
            load(model, model_file, self.device)
            print('loaded ; ' + str(model_file))

        model.eval()
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        scores = []
        for e in range(epoch):
            global_step = 0
            total_loss = 0.0
            total_steps = 0
            examples = []
            iter_bar = tqdm(dataloader, desc="E-{:0=2} : XX.XXXX avg loss ".format(e), position=0)
            for step, batch in enumerate(iter_bar):
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    loss, example = process(batch, model, iter_bar, step)
                examples.append(example)
                if self.num_gpu > 1:
                    loss = loss.mean()

                total_loss += loss.item()
                total_steps += 1
                iter_bar.set_description("E-{:0=2} : {:2.4f} avg loss ".format(e, total_loss / total_steps))

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
        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)
        if model_file is not None:
            load(model, model_file, self.device)
        model.eval()

        if sampler is None:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        predicts = []
        iter_bar = tqdm(dataloader, desc="XX.XXXX avg loss ", position=0)
        for step, batch in enumerate(iter_bar):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                predict = process(batch, model, iter_bar, step)
                predicts.append(predict)
        return predicts

