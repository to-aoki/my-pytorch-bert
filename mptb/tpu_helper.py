# Author Toshihiko Aoki
#
# This file is based on
# https://github.com/pytorch/xla/blob/master/contrib/colab/mnist-training-xrt-1-15.ipynb
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

import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from .utils import set_seeds
from .optimization import get_optimizer, get_scheduler
from .pretrain_dataset import PreTensorPretrainDataset
from .pretrain_tasks import BertPretrainingTasks


def tpu_pretrain(
    model, dataset,
    epochs, batch_size, lr, warmup_proportion, save_dir,
    per_save_epocs, per_save_steps, optimizer_name,
    num_cores=8, log_steps=20, metrics_debug=False,
):

    def train():
        set_seeds(20191206)
        model.train()
        device = xm.xla_device()
        model.to(device)

        max_steps = int(len(dataset) / batch_size * epochs)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = get_optimizer(model=model, lr=lr, optimzier=optimizer_name)
        scheduler = get_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)
        criterion_lm = CrossEntropyLoss(ignore_index=-1)
        if isinstance(model, BertPretrainingTasks):
            criterion_ns = CrossEntropyLoss(ignore_index=-1)
        if isinstance(dataset, PreTensorPretrainDataset) or hasattr(dataset, 'lazy') and dataset.lazy:
            sampler = SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=1)

        def train_loop_fn(per_device_loader):
            tracker = xm.RateTracker()
            model.train()
            for step, batch in enumerate(per_device_loader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask, next_sentence_labels, label_ids = batch
                optimizer.zero_grad()
                masked_lm_logists, auxiliary_logits = model(input_ids, segment_ids, input_mask)

                # Is the operation of the transpose method different??
                masked_lm_loss = criterion_lm(
                    masked_lm_logists.view(-1, label_ids.size(1)).transpose(0, 1), label_ids.view(-1))
                # mlm
                if auxiliary_logits is None:
                    loss = masked_lm_loss
                # nsp
                else:
                    loss = criterion_ns(auxiliary_logits.view(-1, 2), next_sentence_labels.view(-1))
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()

                tracker.add(batch_size)
                if step % log_steps == 0:
                    print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                        xm.get_ordinal(), step, loss.item(), tracker.rate(),
                        tracker.global_rate(), time.asctime()), flush=True)

        for epoch in range(1, epochs + 1):
            p_loader = pl.ParallelLoader(data_loader, [device])
            train_loop_fn(p_loader.per_device_loader(device))
            xm.master_print("Finished training epoch {}".format(epoch))

    def _mp_train_fn(rank, flags):
        torch.set_default_tensor_type('torch.FloatTensor')
        train()

    xmp.spawn(_mp_train_fn, args=(None,), nprocs=num_cores, start_method='fork')

    return
