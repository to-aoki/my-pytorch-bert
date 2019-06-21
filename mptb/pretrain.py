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
"""Bert pre-training."""

import os
from collections import namedtuple
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import RandomSampler

from .models import Config
from .pretrain_tasks import BertPretrainingTasks
from .optimization import get_optimizer
from .pretrain_dataset import PretrainDataset, PreTensorPretrainDataset
from .helper import Helper
from .utils import save, get_logger, get_tokenizer


class BertPretrainier(object):

    def __init__(
        self,
        config_path='../config/bert_base.json',
        max_pos=-1,
        tokenizer=None,
        vocab_path=None,
        sp_model_path=None,
        model_path=None,
        dataset_path=None,
        pretensor_data_path=None,
        pretensor_data_length=-1,
        on_memory=True,
        tokenizer_name='google',
        fp16=False
    ):

        if tokenizer is None and vocab_path is not None:
            self.tokenizer = get_tokenizer(
                vocab_path=vocab_path, sp_model_path=sp_model_path, name=tokenizer_name)
        else:
            self.tokenizer = tokenizer

        if pretensor_data_path is not None:
            self.dataset = PreTensorPretrainDataset(pretensor_data_path, pretensor_data_length)
        elif dataset_path is not None:
            self.dataset = self.get_dataset(dataset_path, self.tokenizer, max_pos=max_pos, on_memory=on_memory)
            max_pos = self.dataset.max_pos

        config = Config.from_json(config_path, len(self.tokenizer), max_pos)
        print(config)
        self.max_pos = max_pos
        self.model = BertPretrainingTasks(config)
        self.helper = Helper(fp16=fp16)
        self.helper.set_model(self.model)
        self.model_path = model_path
        super().__init__()

    def get_dataset(
        self,
        dataset_path,
        tokenizer,
        max_pos=-1,
        on_memory=True,
    ):
        if hasattr(self, 'max_pos'):
            max_pos = self.max_pos

        if max_pos < 5:
            # max_pos = statistics.median(all-sentence-tokens)
            import statistics
            with open(dataset_path, 'r', newline="\n", encoding="utf-8") as data:
                tokens = list(map(tokenizer.tokenize, data.readlines()))
                median_pos = round(statistics.median(list(map(lambda x: len(x), tokens))))
            max_pos = median_pos * 2 + 3  # [CLS]a[SEP]b[SEP]
            print("max_pos (median):", max_pos)

        return PretrainDataset(
            tokenizer=tokenizer, max_pos=max_pos, dataset_path=dataset_path, on_memory=on_memory
        )

    def train(
        self,
        dataset=None,
        tokenizer=None,
        sampler=None,
        traing_model_path=None,
        batch_size=4,
        epochs=20,
        lr=5e-5,
        warmup_proportion=0.1,
        save_dir='../pretrain/',
        per_save_epochs=1,
        is_save_after_training=True
    ):

        if tokenizer is None and hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer
        else:
            raise ValueError('require tokenizer.')

        if dataset is None:
            if hasattr(self, 'dataset'):
                dataset = self.dataset
            else:
                raise ValueError('require dataset')

        if sampler is None:
            sampler = RandomSampler(dataset)

        max_steps = int(len(dataset) / batch_size * epochs)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = get_optimizer(
            model=self.model, lr=lr, warmup_steps=warmup_steps, max_steps=max_steps, fp16=self.helper.fp16)
        if self.model_path is not None and self.model_path != '':
            self.helper.load_model(self.model, self.model_path, optimizer)
        criterion_lm = CrossEntropyLoss(ignore_index=-1, reduction='none')
        criterion_ns = CrossEntropyLoss(ignore_index=-1)

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, next_sentence_labels, label_ids = batch
            masked_lm_logists, next_sentence_logits = model(input_ids, segment_ids, input_mask)
            lm_labels = label_ids.view(-1)
            numerator = criterion_lm(masked_lm_logists.view(-1, len(tokenizer)), lm_labels)
            masked_lm_loss = numerator.sum() / (len(lm_labels) + 1e-5)
            next_sentence_loss = criterion_ns(next_sentence_logits.view(-1, 2), next_sentence_labels.view(-1))
            return masked_lm_loss + next_sentence_loss

        if self.helper.fp16:
            def adjustment_every_step(model, dataset, loss, global_step, optimizer):
                from mptb.optimization import update_lr_apex
                update_lr_apex(optimizer, global_step, lr, warmup_steps, max_steps)
        else:
            def adjustment_every_step(model, dataset, total_loss, total_steps, optimizer):
                pass

        def adjustment_every_epoch(model, dataset, total_loss, total_steps, optimizer):
            exit(0)

        loss = self.helper.train(
            process=process,
            model=self.model,
            dataset=dataset,
            sampler=sampler,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            model_file=traing_model_path,
            save_dir=save_dir,
            per_save_epochs=per_save_epochs,
            adjustment_every_epoch=None,
            adjustment_every_step=adjustment_every_step
        )
        self.learned = True

        if is_save_after_training:
            output_model_path = os.path.join(save_dir, "bert.pt")
            save(self.model.bert, output_model_path)
        return loss

    def evaluate(
        self,
        dataset=None,
        tokenizer=None,
        sampler=None,
        model_path=None,
        batch_size=2,
        is_reports_output=False,
        log_dir=None
    ):

        if model_path is None and hasattr(self, 'model_path'):
            model_path = self.model_path
            self.learned = True

        if model_path is not None:
            self.helper.load_model(self.model, model_path)

        if not hasattr(self, 'learned') or not hasattr(self, 'model'):
            raise ValueError('not learning model.')

        if tokenizer is None and hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer
        else:
            raise ValueError('require tokenizer.')

        if dataset is None:
            if hasattr(self, 'dataset'):
                dataset = self.dataset
            else:
                raise ValueError('require dataset')

        if sampler is None:
            sampler = RandomSampler(dataset)

        criterion_lm = NLLLoss(ignore_index=-1, reduction='none')
        criterion_ns = NLLLoss(ignore_index=-1)
        Example = namedtuple('Example', ('lm_pred', 'lm_true', 'ns_pred', 'ns_true'))

        def process(batch, model, iter_bar, step):
            input_ids, segment_ids, input_mask, next_sentence_labels, label_ids = batch
            masked_lm_loss, next_sentence_loss = model(input_ids, segment_ids, input_mask)

            masked_lm_probs = F.log_softmax(masked_lm_loss.view(-1, len(tokenizer)), 1)
            masked_lm_predictions = masked_lm_probs.max(1, False)
            lm_labels = label_ids.view(-1)

            ns_probs = F.log_softmax(next_sentence_loss.view(-1, 2), 1)
            ns_predictions = ns_probs.max(1, False)
            ns_labels = next_sentence_labels.view(-1)

            example = Example(
                masked_lm_predictions[1].tolist(), lm_labels.tolist(),
                ns_predictions[1].tolist(), ns_labels.tolist())

            masked_lm_loss = criterion_lm(masked_lm_probs, lm_labels)
            masked_lm_loss = masked_lm_loss.sum()/(len(lm_labels) + 1e-5)
            next_sentence_loss = criterion_ns(ns_probs, ns_labels)

            return masked_lm_loss + next_sentence_loss, example

        if not is_reports_output:
            return self.helper.evaluate(process, self.model, dataset, sampler, batch_size, model_path)

        logger = None
        if log_dir is not None and log_dir is not '':
            logger = get_logger('eval', log_dir, False)

        def example_reports(examples):
            if examples is None or len(examples) is 0:
                return
            try:
                from sklearn.metrics import classification_report
                from sklearn.exceptions import UndefinedMetricWarning
                import warnings
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            except ImportError:
                import warnings
                warnings.warn('sklearn.metrics.classification_report fail to import', ImportWarning)
                return

            y_lm_preds, y_lm_trues, y_ns_preds, y_ns_trues = [], [], [], []
            for lm_pred, lm_true, ns_pred, ns_true in examples:
                for p in lm_pred:
                    y_lm_preds.append(p)
                for t in lm_true:
                    y_lm_trues.append(t)
                for p in ns_pred:
                    y_ns_preds.append(p)
                for t in ns_true:
                    y_ns_trues.append(t)

            if logger is not None:
                lm_reports = classification_report(y_lm_trues, y_lm_preds, output_dict=True)
                for k, v in lm_reports.items():
                    for ck, cv in v.items():
                        logger.info(str(k) + "," + str(ck) + "," + str(cv))

                ns_reports = classification_report(y_ns_trues, y_ns_preds, output_dict=True)
                for k, v in ns_reports.items():
                    for ck, cv in v.items():
                        logger.info(str(k) + "," + str(ck) + "," + str(cv))
            else:
                print(classification_report(y_lm_trues, y_lm_preds))
                print(classification_report(y_ns_trues, y_ns_preds))

        return self.helper.evaluate(process, self.model, dataset, sampler, batch_size, model_path, example_reports)
