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
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seeds, save, load, get_device


class Helper(object):

    def __init__(self, seeds=20151106, device=None):
        super().__init__()
        set_seeds(seeds)
        self.num_gpu = 0

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(get_device())

        if torch.cuda.is_available() and self.device is not 'cpu':
            self.num_gpu = torch.cuda.device_count()

        print("device: {} num_gpu: {}".format(self.device, self.num_gpu))

    def training(
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
        epoch_dataset_adjust=None
    ):

        model.to(self.device)
        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)

        if model_file is not None and model_file is not '':
            # warmup_steps over-ride
            load(model, model_file, self.device, optimizer)

        global_step = optimizer.get_step()
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        model.train()
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
                loss.backward()
                total_loss += loss.item()
                total_steps += 1
                iter_bar.set_description("E-{:0=2} : {:2.4f} avg loss ".format(e, total_loss / total_steps))
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if per_save_epoc > 0 and (e + 1) % per_save_epoc is 0:
                output_model_file = os.path.join(save_dir, "train_model_" + str(e) + "_" + str(global_step) + ".pt")
                save(model, output_model_file, optimizer)

            if epoch_dataset_adjust is not None:
                epoch_dataset_adjust(dataset)

    def evaluate(
        self,
        process,
        model,
        dataset,
        sampler,
        batch_size,
        model_file,
        examples_reports=None
    ):

        model.to(self.device)
        if self.num_gpu > 1:
            model = torch.nn.DataParallel(model)

        if model_file is not None:
            load(model, model_file, self.device)

        global_step = 0

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        model.eval()
        total_loss = 0.0
        total_steps = 0
        examples = []
        iter_bar = tqdm(dataloader, desc="XX.XXXX avg loss ", position=0)
        for step, batch in enumerate(iter_bar):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                loss, example = process(batch, model, iter_bar, step)
            examples.append(example)
            if self.num_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            total_steps += 1
            iter_bar.set_description("{:2.4f} avg loss ".format(total_loss / total_steps))
            global_step += 1

        if examples_reports is not None:
            examples_reports(examples)

