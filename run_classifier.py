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
"""Bert classifier."""

import os
from collections import namedtuple
import models
import optimization
import tokenization_sentencepiece
import tokenization
import torch
from torch.nn import CrossEntropyLoss, NLLLoss

from finetuning import Classifier
from class_csv_dataset import BertCsvDataset
from helper import Helper
from utils import save, load, get_logger, make_balanced_classes_weights


def classification(
    config_path='config/bert_base.json',
    dataset_path='data/livedoor_train_class.tsv',
    pretrain_path='collections/bert-wiki-ja.pt',
    model_path=None,
    vocab_path='collections/wiki-ja.vocab',
    sp_model_path='collections/wiki-ja.model',
    save_dir='classifier/',
    log_dir=None,
    batch_size=2,
    max_pos=512,
    lr=5e-5,
    warmup_proportion=0.1,  # warmup_steps = len(dataset) / batch_size * epoch * warmup_proportion
    epoch=5,
    per_save_epoc=1,
    mode='train',
    label_num=None,
    balanced=False,
    read_head=False
):
    if sp_model_path is not None:
        tokenizer = tokenization_sentencepiece.FullTokenizer(
            sp_model_path, vocab_path, do_lower_case=True)
    else:
        tokenizer = tokenization.FullTokenizer(vocab_path, do_lower_case=True)

    config = models.Config.from_json(config_path, len(tokenizer), max_pos)
    dataset = BertCsvDataset(dataset_path, tokenizer, max_pos, label_num, header_skip=not read_head)

    model = Classifier(config, label_num)

    print('model params :', config)
    helper = Helper()
    if mode == 'train':
        logger = get_logger('eval', log_dir, True)
        if pretrain_path is not None:
            load(model.bert, pretrain_path)

        max_steps = int(len(dataset) / batch_size * epoch)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = optimization.get_optimizer(model, lr, warmup_steps, max_steps)

        balance_weights = None
        if balanced:
            balance_weights = torch.Tensor(make_balanced_classes_weights(dataset.per_label_records_num))
        criterion = CrossEntropyLoss(weight=balance_weights)

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            return criterion(logits.view(-1, label_num), label_id.view(-1))

        helper.training(process, model, dataset, optimizer, batch_size, epoch, model_path, save_dir, per_save_epoc)

        output_model_path = os.path.join(save_dir, "classifier.pt")
        save(model, output_model_path)

    elif mode == 'eval':

        criterion = CrossEntropyLoss()
        Example = namedtuple('Example', ('pred', 'true'))
        logger = None
        if log_dir is not None and log_dir is not '':
            logger = get_logger('eval', log_dir, False)

        def process(batch, model, iter_bar, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, label_num), label_id.view(-1))
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

            if logger is not None:
                classify_reports = classification_report(y_trues, y_preds, output_dict=True)
                for k, v in classify_reports.items():
                    for ck, cv in v.items():
                        logger.info(str(k) + "," + str(ck) + "," + str(cv))
            else:
                print(classification_report(y_trues, y_preds))

        helper.evaluate(process, model, dataset, batch_size, model_path, example_reports)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BERT classification.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--dataset_path', help='Dataset file (TSV file) path for classification.', required=True,
                        type=str)
    parser.add_argument('--pretrain_path', help='Pre-training PyTorch model path.', nargs='?',
                        type=str, default='bert-wiki-ja/bert-wiki-ja.pt')
    parser.add_argument('--model_path', help='Classifier PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.', nargs='?',
                        type=str, default='bert-wiki-ja/wiki-ja.vocab')
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default='bert-wiki-ja/wiki-ja.model')
    parser.add_argument('--save_dir', help='Classification model saving directory path.', nargs='?',
                        type=str, default='classifier/')
    parser.add_argument('--log_dir', help='Logging file path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--batch_size',  help='Batch size', nargs='?',
                        type=int, default=4)
    parser.add_argument('--max_pos', help='The maximum sequence length for BERT (slow as big).', nargs='?',
                        type=int, default=512)
    parser.add_argument('--lr', help='Learning rate', nargs='?',
                        type=float, default=2e-5)
    parser.add_argument('--warmup_steps', help='Warm-up steps proportion.', nargs='?',
                        type=float, default=0.1)
    parser.add_argument('--epoch', help='Epoch', nargs='?',
                        type=int, default=10)
    parser.add_argument('--per_save_epoc', help=
                        'Saving training model timing is the number divided by the epoch number', nargs='?',
                        type=int, default=1)
    parser.add_argument('--mode', help='train or eval', nargs='?',
                        type=str, default='train')
    parser.add_argument('--label_num', help='labels number', required=True,
                        type=int)
    parser.add_argument('--balanced', action='store_true',
                        help='Use automatically adjust weights')
    parser.add_argument('--read_head', action='store_true',
                        help='Use not include header TSV file')
#    parser.add_argument('--labels', nargs='+', help='<Required> labels', required=True)

    args = parser.parse_args()
    classification(args.config_path, args.dataset_path, args.pretrain_path, args.model_path, args.vocab_path,
                   args.sp_model_path, args.save_dir, args.log_dir, args.batch_size, args.max_pos, args.lr,
                   args.warmup_steps, args.epoch, args.per_save_epoc, args.mode, args.label_num,
                   args.balanced, args.read_head)
