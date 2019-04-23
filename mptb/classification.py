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
"""Bert Classification"""

import os
from collections import namedtuple
from . models import Config
from . optimization import get_optimizer
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, WeightedRandomSampler

from .finetuning import Classifier
from .class_dataset import ClassDataset
from .helper import Helper
from .utils import save, load, get_logger, make_balanced_classes_weights, get_tokenizer


class BertClassifierEstimator(object):

    def __init__(
        self,
        config_path='../config/bert_base.json',
        max_pos=-1,
        tokenizer=None,
        vocab_path=None,
        sp_model_path=None,
        model_path=None,
        pretrain_path=None,
        dataset_path=None,
        header_skip=True,
        label_num=-1,
        hidden_size=-1,
        tokenizer_name='google',
        under_sampling=False
    ):
        if tokenizer is None:
            self.tokenizer = get_tokenizer(
                vocab_path=vocab_path, sp_model_path=sp_model_path, name=tokenizer_name)
        else:
            self.tokenizer = tokenizer

        config = Config.from_json(config_path, len(self.tokenizer), max_pos)
        self.max_pos = config.max_position_embeddings
        print(config)
        if dataset_path is not None:
            self.dataset = self.get_dataset(
                self.tokenizer,
                dataset_path,
                header_skip=header_skip, under_sampling=under_sampling)
            label_num = self.dataset.label_num()

        if label_num < 0:
            raise ValueError('label_num require positive value.')

        self.model = Classifier(config, label_num=label_num, hidden_size=hidden_size)
        if model_path is None and pretrain_path is not None:
            load(self.model.bert, pretrain_path)
            print('pretain model loaded: ' + pretrain_path)
            self.pretrain = True

        self.helper = Helper()
        if model_path is not None and model_path != '':
            self.model_path = model_path
            self.helper.load_model(self.model, model_path)
            self.model.eval()
            self.learned = True
        super().__init__()

    def get_dataset(
        self,
        tokenizer=None,
        dataset_path=None, header_skip=True,
        sentence_a=[], sentence_b=[], labels=[],
        under_sampling=False
    ):
        if tokenizer is None and hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer

        if tokenizer is None:
            raise ValueError('dataset require tokenizer')

        return ClassDataset(
            tokenizer=tokenizer, max_pos=self.max_pos, dataset_path=dataset_path, header_skip=header_skip,
            sentence_a=sentence_a, sentence_b=sentence_b, labels=labels,
            under_sampling=under_sampling
        )

    @staticmethod
    def get_class_balanced_sampler(
        dataset
    ):
        assert isinstance(dataset, ClassDataset), 'dataset is an instance of ClassDataset.'
        indices = list(range(len(dataset)))
        num_samples = len(dataset)
        weights = [1.0 / dataset.per_label_records_num[dataset[index][3].item()] for index in indices]
        weights = torch.tensor(weights)
        return WeightedRandomSampler(weights, num_samples)

    def train(
        self,
        dataset=None,
        sampler=None,
        traing_model_path=None,
        pretrain_path=None,
        model_path=None,
        batch_size=4,
        epoch=10,
        lr=5e-5,
        warmup_proportion=0.1,
        balance_weight=False,
        balance_sample=False,
        under_sampling_cycle=False,
        save_dir='../classifier/',
        per_save_epoch=1,
        is_save_after_training=True
    ):

        if dataset is None:
            if hasattr(self, 'dataset'):
                dataset = self.dataset
            else:
                raise ValueError('require dataset')

        if traing_model_path is None and hasattr(self, 'model_path'):
            model_path = self.model_path

        if not hasattr(dataset,'sampling_index'):
            under_sampling_cycle = False

        if model_path is None and pretrain_path is not None:
            load(self.model.bert, pretrain_path)
            self.pretrain = True

        if not hasattr(self, 'pretrain') and not hasattr(self, 'learned'):
            raise ValueError('require pretrain model')

        max_steps = int(len(dataset) / batch_size * epoch)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = get_optimizer(self.model, lr, warmup_steps, max_steps)

        balance_weights = None
        if balance_weight:
            balance_weights = torch.tensor(
                make_balanced_classes_weights(
                    dataset.per_label_records_num), device=self.helper.device)

        criterion = CrossEntropyLoss(weight=balance_weights)

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, self.model.label_len), label_id.view(-1))
            return loss

        if sampler is None:
            if balance_sample:
                sampler = BertClassifierEstimator.get_class_balanced_sampler(dataset)
            else:
                sampler = RandomSampler(dataset)

        def epoch_dataset_adjust(dataset):
            if under_sampling_cycle:
                dataset.next_under_samples()
            else:
                pass

        loss = self.helper.train(
            process, self.model, dataset, sampler, optimizer, batch_size, epoch, traing_model_path, save_dir,
                per_save_epoch, epoch_dataset_adjust)
        self.learned = True
        if is_save_after_training:
            output_model_path = os.path.join(save_dir, "classifier.pt")
            save(self.model, output_model_path)

        return loss

    def evaluate(
        self,
        dataset=None,
        sampler=None,
        model_path=None,
        batch_size=2,
        epoch=1,
        is_reports_output=True,
        log_dir=None
    ):

        if model_path is None and hasattr(self, 'model_path'):
            model_path = self.model_path
            self.learned = True

        if not hasattr(self, 'learned') or not hasattr(self, 'model'):
            raise ValueError('not learning model.')

        if dataset is None:
            if hasattr(self, 'dataset'):
                dataset = self.dataset
            else:
                raise ValueError('require dataset')

        if sampler is None:
            sampler = RandomSampler(dataset)

        criterion = CrossEntropyLoss()

        Example = namedtuple('Example', ('pred', 'true'))
        logger = None
        if log_dir is not None and log_dir is not '':
            logger = get_logger('eval', log_dir, False)

        def process(batch, model, iter_bar, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, self.model.label_len), label_id.view(-1))
            _, label_pred = logits.max(1)
            example = Example(label_pred.tolist(), label_id.tolist())
            return loss, example

        def compute_epoch_score(examples):
            accuracy = 0.
            for preds, trues in examples:
                if preds == trues:
                    accuracy += 1
            return accuracy/len(examples)

        if not is_reports_output:
            return self.helper.evaluate(
                process, self.model, dataset, sampler, batch_size, model_path,
                    compute_epoch_score=compute_epoch_score, epoch=epoch)

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
                warnings.warn('sklearn.metrics.classification_report failed to import', ImportWarning)
                return

            y_preds, y_trues = [], []
            for preds, trues in examples:
                for p in preds:
                    y_preds.append(p)
                for t in trues:
                    y_trues.append(t)

            if logger is not None:
                classify_reports = classification_report(y_trues, y_preds, output_dict=True)
                for k, v in classify_reports.items():
                    for ck, cv in v.items():
                        logger.info(str(k) + "," + str(ck) + "," + str(cv))
            else:
                print(classification_report(y_trues, y_preds))

        return self.helper.evaluate(
            process, self.model, dataset, sampler, batch_size, model_path, example_reports,
            compute_epoch_score=compute_epoch_score, epoch=epoch)

    def predict(
        self,
        dataset=None,
        sampler=None,
        model_path=None,
        batch_size=1
    ):

        if model_path is None and hasattr(self, 'model_path'):
            model_path = self.model_path
            self.learned = True

        if not hasattr(self, 'learned') or not hasattr(self, 'model'):
            raise ValueError('not learning model.')

        if dataset is None:
            if hasattr(self, 'dataset'):
                dataset = self.dataset
            else:
                raise ValueError('require dataset')

        def process(batch, model, iter_bar, step):
            input_ids, segment_ids, input_mask = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            return label_pred.tolist()

        return self.helper.predict(
            process, self.model, dataset, sampler, batch_size, model_path)