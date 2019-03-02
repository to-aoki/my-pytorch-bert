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

import os
from collections import namedtuple
import models
import optimization
import tokenization_sentencepiece
import tokenization
from torch.nn import CrossEntropyLoss, NLLLoss

from finetuning import Classifier
from class_csv_dataset import BertCsvDataset
from helper import Helper
from utils import save, load, get_logger


def main(
    model_cfg='config/bert_base.json',
    file_path='livedoor/livedoor_train_class.tsv',
    pretrain_file='pretrrain/multi_caused_L-12_H-768_A-12.pt',
    model_file=None,
    vocab_file='tests/sample_text.vocab',
    sp_model_file='tests/sample_text.model',
    save_dir='classifier/',
    log_dir='logs/',
    batch_size=2,
    max_len=128,
    lr=2e-5,
    warmup_proportion=0.1,  # warmup_steps = len(dataset) / batch_size * epoch * warmup_proportion
    epoch=5,
    per_save_epoc=1,
    label_num=9,
    mode='train'):

    if sp_model_file is not None:
        tokenizer = tokenization_sentencepiece.FullTokenizer(
            sp_model_file, vocab_file, do_lower_case=True)
    else:
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    config = models.Config.from_json(model_cfg, len(tokenizer), max_len)
    dataset = BertCsvDataset(file_path, tokenizer, max_len, label_num)

    model = Classifier(config, label_num)

    print('model params :', config)
    helper = Helper()

    if mode is 'train':
        logger = get_logger('eval', log_dir, True)
        if pretrain_file is not None:
            load(model.bert, pretrain_file)

        warmup_steps = int(len(dataset) / batch_size * epoch * warmup_proportion)
        optimizer = optimization.get_optimizer(model, lr, warmup_steps)
        criterion = CrossEntropyLoss()

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            return criterion(logits, label_id)

        helper.training(process, model, dataset, optimizer, batch_size, epoch, model_file, save_dir, per_save_epoc)

        output_model_file = os.path.join(save_dir, "classifier.pt")
        save(model, output_model_file)

    elif mode is 'eval':
        logger = get_logger('eval', log_dir, False)
        criterion = CrossEntropyLoss()

        Example = namedtuple('Example', ('pred', 'true'))

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            _, label_pred = logits.max(1)
            example = Example(label_pred.tolist(), label_id.tolist())
            return loss, example

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

            classify_reports = classification_report(y_trues, y_preds, output_dict=True)
            for k, v in classify_reports.items():
                for ck, cv in v.items():
                    logger.info(str(k) + "," + str(ck) + "," + str(cv))

        helper.evaluate(process, model, dataset, batch_size, epoch, model_file, example_reports)

main()
