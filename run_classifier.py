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
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, WeightedRandomSampler

from finetuning import Classifier
from class_csv_dataset import BertCsvDataset
from helper import Helper
from utils import save, load, get_logger, make_balanced_classes_weights, default_preprocessor


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
    per_save_epoch=1,
    mode='train',
    label_num=None,
    balance_weight=False,
    balance_sample=False,
    under_sampling=False,
    under_sampling_cycle=False,
    use_mecab=False,
    use_jumanapp=False,
    read_head=False
):

    preprocessor = default_preprocessor

    if sp_model_path is not None:
        tokenizer = tokenization_sentencepiece.FullTokenizer(
            sp_model_path, vocab_path, preprocessor=preprocessor)
    else:
        if use_mecab:
            import tokenization_mecab
            tokenizer = tokenization_mecab.FullTokenizer(vocab_path)
        elif use_jumanapp:
            tokenizer = tokenization.FullTokenizer(vocab_path, preprocessor=preprocessor)
        else:
            tokenizer = tokenization.FullTokenizer(vocab_path, preprocessor=preprocessor)

    config = models.Config.from_json(config_path, len(tokenizer), max_pos)

    if under_sampling_cycle:
        under_sampling = True

    dataset = BertCsvDataset(tokenizer, max_pos, label_num, dataset_path,
                             under_sampling=under_sampling,
                             header_skip=not read_head)

    model = Classifier(config, label_num)

    print('model params :', config)
    helper = Helper()
    if mode == 'train':

        if model_path is None and pretrain_path is not None:
            load(model.bert, pretrain_path)

        max_steps = int(len(dataset) / batch_size * epoch)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = optimization.get_optimizer(model, lr, warmup_steps, max_steps)

        balance_weights = None
        if balance_weight:
            balance_weights = torch.tensor(
                make_balanced_classes_weights(dataset.per_label_records_num), device=helper.device)

        criterion = CrossEntropyLoss(weight=balance_weights)

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, label_num), label_id.view(-1))
            return loss

        if balance_sample:
            indices = list(range(len(dataset)))
            num_samples = len(dataset)
            weights = [1.0 / dataset.per_label_records_num[dataset[index][3].item()] for index in indices]
            weights = torch.tensor(weights)
            sampler = WeightedRandomSampler(weights, num_samples)
        else:
            sampler = RandomSampler(dataset)

        def epoch_dataset_adjust(dataset):
            if under_sampling_cycle:
                dataset.next_under_samples()
            else:
                pass

        helper.training(process, model, dataset, sampler, optimizer, batch_size, epoch, model_path, save_dir,
                        per_save_epoch, epoch_dataset_adjust)

        name, _ = os.path.splitext(os.path.basename(dataset_path))
        output_model_path = os.path.join(save_dir, name + "_classifier.pt")
        save(model, output_model_path)

    elif mode == 'eval':

        sampler = RandomSampler(dataset)
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

        helper.evaluate(process, model, dataset, sampler, batch_size, model_path, example_reports)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BERT classification.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--dataset_path', help='Dataset file (TSV file) path for classification.', required=True,
                        type=str)
    parser.add_argument('--pretrain_path', help='Pre-training PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--model_path', help='Classifier PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.', nargs='?', required=True,
                        type=str)
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default=None)
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
    parser.add_argument('--per_save_epoch', help=
                        'Saving training model timing is the number divided by the epoch number', nargs='?',
                        type=int, default=1)
    parser.add_argument('--mode', help='train or eval', nargs='?',
                        type=str, default='train')
    parser.add_argument('--label_num', help='labels number', required=True,
                        type=int)
    parser.add_argument('--balance_weight', action='store_true',
                        help='Use automatically adjust weights')
    parser.add_argument('--balance_sample', action='store_true',
                        help='Use automatically adjust samples(random)')
    parser.add_argument('--under_sampling', action='store_true',
                        help='Use automatically adjust under samples')
    parser.add_argument('--under_sampling_cycle', action='store_true',
                        help='Use automatically adjust under samples cycle peer')
    parser.add_argument('--use_mecab', action='store_true',
                        help='Use Mecab Tokenizer')
    parser.add_argument('--use_jumanpp', action='store_true',
                        help='Use Juman++(v2.0.0) Morpheme tokenized text')
    parser.add_argument('--read_head', action='store_true',
                        help='Use not include header TSV file')

    args = parser.parse_args()
    classification(args.config_path, args.dataset_path, args.pretrain_path, args.model_path, args.vocab_path,
                   args.sp_model_path, args.save_dir, args.log_dir, args.batch_size, args.max_pos, args.lr,
                   args.warmup_steps, args.epoch, args.per_save_epoch, args.mode, args.label_num,
                   args.balance_weight, args.balance_sample, args.under_sampling, args.under_sampling_cycle,
                   args.use_mecab, args.use_jumanpp, args.read_head)
