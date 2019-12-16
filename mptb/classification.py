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
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, WeightedRandomSampler

from .bert import Config
from .optimization import get_optimizer, get_scheduler
from .finetuning import Classification, MultipleChoice
from .class_dataset import ClassDataset
from .choice_dataset import SwagDataset
from .helper import Helper
from .utils import save, get_logger, make_balanced_classes_weights, \
    get_tokenizer, load_from_google_bert_model, AttributeDict


class BertClassifier(object):

    def __init__(
        self,
        config_path='../config/bert_base.json',
        max_pos=-1,
        tokenizer=None,
        vocab_path=None,
        sp_model_path=None,
        model_path=None,
        pretrain_path=None,
        tf_pretrain_path=None,
        dataset_path=None,
        header_skip=True,
        label_num=-1,
        bert_layer_num=-1,
        tokenizer_name='google',
        under_sampling=False,
        fp16=False,
        task='class',
        device=None,
        quantize=False,
        model_name='bert',
        encoder_json_path=None,
        vocab_bpe_path=None,
    ):
        if tokenizer is None:
            self.tokenizer = get_tokenizer(
                vocab_path=vocab_path, sp_model_path=sp_model_path,
                encoder_json_path=encoder_json_path, vocab_bpe_path=vocab_bpe_path, name=tokenizer_name)
        else:
            self.tokenizer = tokenizer

        if quantize:
            with open(config_path, "r", encoding="UTF-8") as reader:
                import json
                config_dict = json.load(reader)
                config = AttributeDict(config_dict)
        else:
            config = Config.from_json(config_path, len(self.tokenizer), max_pos, bert_layer_num)
        print(config)
        self.max_pos = config.max_position_embeddings
        self.task = task
        if dataset_path is not None:
            self.dataset = self.get_dataset(
                self.tokenizer, dataset_path, header_skip=header_skip, under_sampling=under_sampling)
            if self.task == 'choice':
                self.model = MultipleChoice(config, model_name=model_name)
            else:
                if label_num != -1 and label_num != self.dataset.label_num():
                    raise ValueError(
                        'label num mismatch. input : {} datset : {}'.format(label_num, self.dataset.label_num()))
                if quantize:
                    print('quantized classification model')
                    from .quantized_bert import QuantizedBertForSequenceClassification
                    self.model = QuantizedBertForSequenceClassification(config, num_labels=self.dataset.label_num())
                else:
                    self.model = Classification(config, num_labels=self.dataset.label_num(), model_name=model_name)

        self.pretrain = False
        self.helper = Helper(device=device, fp16=fp16)

        if model_path is None and pretrain_path is not None:
            self.helper.load_model(model=self.model.bert, model_path=pretrain_path, strict=not quantize)
            print('pretain model loaded: ' + pretrain_path)
            self.pretrain = True
        if not hasattr(self, 'pretrain') and tf_pretrain_path is not None:
            load_from_google_bert_model(self.model.bert, tf_pretrain_path)
            print('pretain model loaded: ' + tf_pretrain_path)
            self.pretrain = True

        if model_path is not None and model_path != '':
            self.model_path = model_path
            self.helper.load_model(self.model, model_path)
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

        if self.task == 'choice':
            return SwagDataset(
                tokenizer=tokenizer, max_pos=self.max_pos, dataset_path=dataset_path, header_skip=header_skip
            )

        else:
            dataset = ClassDataset(
                tokenizer=tokenizer, max_pos=self.max_pos, dataset_path=dataset_path, header_skip=header_skip,
                sentence_a=sentence_a, sentence_b=sentence_b, labels=labels,
                under_sampling=under_sampling
            )
            if dataset.label_num() < 0:
                raise ValueError('label_num require positive value.')
            return dataset

    @staticmethod
    def get_class_balanced_sampler(dataset):
        if not hasattr(dataset, 'per_label_records_num'):
            return RandomSampler(dataset)
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
        tf_pretrain_path=None,
        batch_size=4,
        epochs=10,
        lr=5e-5,
        warmup_proportion=0.1,
        balance_weight=False,
        balance_sample=False,
        under_sampling_cycle=False,
        save_dir='../classifier/',
        per_save_epochs=1,
        is_save_after_training=True,
        optimizer_name='bert'
    ):

        if dataset is None:
            if hasattr(self, 'dataset'):
                dataset = self.dataset
            else:
                raise ValueError('require dataset')

        is_model_laoded = False
        if traing_model_path is None and hasattr(self, 'model_path'):
            traing_model_path = self.model_path
            is_model_laoded = True

        if not hasattr(dataset, 'sampling_index'):
            under_sampling_cycle = False

        if not is_model_laoded and not self.pretrain:
            if pretrain_path is not None:
                self.helper.load_model(model=self.model.bert, model_path=pretrain_path, strict=not quantize)
                print('pretain model loaded: ' + pretrain_path)
                self.pretrain = True
            elif tf_pretrain_path is not None:
                load_from_google_bert_model(self.model.bert, tf_pretrain_path)
                print('pretain model loaded: ' + tf_pretrain_path)
                self.pretrain = True

        if not is_model_laoded and not hasattr(self, 'pretrain') and not hasattr(self, 'learned'):
            raise ValueError('require model')

        max_steps = int(len(dataset) / batch_size * epochs)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = get_optimizer(model=self.model, lr=lr, optimizer=optimizer_name)
        scheduler = get_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)

        balance_weights = None
        if balance_weight:
            balance_weights = torch.tensor(
                make_balanced_classes_weights(
                    dataset.per_label_records_num), device=self.helper.device)

        criterion = CrossEntropyLoss(weight=balance_weights)

        def process(batch, model, iter_bar, epochs, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            if hasattr(model, 'label_len'):
                loss = criterion(logits.view(-1, model.label_len), label_id.view(-1))
            else:
                loss = criterion(logits.view(-1, input_ids.shape[1]), label_id.view(-1))
            return loss

        def adjustment_every_step(model, dataset, loss, total_steps, global_step, optimizer, batch_size):
            pass

        if sampler is None:
            if balance_sample and self.task != 'choice':
                sampler = BertClassifier.get_class_balanced_sampler(dataset)
            else:
                sampler = RandomSampler(dataset)

        if under_sampling_cycle:
            def adjustment_every_epoch(model, dataset, total_loss, total_steps, optimizer):
                dataset.next_under_samples()
        else:
            def adjustment_every_epoch(model, dataset, total_loss, total_steps, optimizer):
                pass

        loss = self.helper.train(
            process=process,
            model=self.model,
            dataset=dataset,
            sampler=sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            epochs=epochs,
            model_file=traing_model_path,
            save_dir=save_dir,
            per_save_epochs=per_save_epochs,
            adjustment_every_epoch=adjustment_every_epoch,
            adjustment_every_step=adjustment_every_step
        )
        self.learned = True
        if is_save_after_training:
            output_model_path = os.path.join(save_dir, "classifier.pt")
            if self.helper.num_gpu > 1:
                save(self.model.module, output_model_path)
            else:
                save(self.model, output_model_path)

        return loss

    def evaluate(
        self,
        dataset=None,
        sampler=None,
        model_path=None,
        batch_size=2,
        epochs=1,
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
            if hasattr(model, 'label_len'):
                loss = criterion(logits.view(-1, self.model.label_len), label_id.view(-1))
            else:
                loss = criterion(logits.view(-1, input_ids.shape[1]), label_id.view(-1))
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
                compute_epoch_score=compute_epoch_score, epochs=epochs)

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
            compute_epoch_score=compute_epoch_score, epochs=epochs)

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

        def process(batch, model, step):
            input_ids, segment_ids, input_mask = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            return label_pred.tolist()

        return self.helper.predict(
            process, self.model, dataset, sampler, batch_size, model_path)
