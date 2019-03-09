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

import models 
import pretrain_tasks
import optimization
from pretrain_dataset import PretrainDataset
import tokenization_sentencepiece
import tokenization
from helper import Helper
from utils import save, get_logger


def bert_pretraining(
        config_path='config/bert_base.json',
        dataset_path='tests/sample_text.txt',
        model_path=None,
        vocal_path='tests/sample_text.vocab',
        sp_model_path='tests/sample_text.model',
        save_dir='pretrain/',
        log_dir='logs/',
        batch_size=2,
        max_pos=128,
        lr=5e-5,
        warmup_proportion=0.1,  # warmup_steps = len(dataset) / batch_size * epoch * warmup_proportion
        epoch=5,
        per_save_epoc=1,
        mode='train'):

    assert mode is not None and (mode == 'train' or mode == 'eval'), 'support mode train or eval.'

    if sp_model_path is not None:
        tokenizer = tokenization_sentencepiece.FullTokenizer(
            sp_model_path, vocal_path, do_lower_case=True)
    else:
        tokenizer = tokenization.FullTokenizer(vocal_path, do_lower_case=True)

    if max_pos is None:
        # max_pos = statistics.median(all-sentence-tokens)
        import statistics
        with open(dataset_path, 'r', newline="\n", encoding="utf-8") as data:
            tokens = list(map(tokenizer.tokenize, data.readlines()))
            max_pos = round(statistics.median(list(map(lambda x: len(x), tokens))))
        max_pos = max_pos*2+3  # [CLS]a[SEP]b[SEP]
        print("max_pos (median):", max_pos)

    train_dataset = PretrainDataset(
        dataset_path,
        tokenizer,
        max_pos=max_pos,
        corpus_lines=None,
        on_memory=True
    )

    config = models.Config.from_json(config_path, len(tokenizer), max_pos)
    print('model params :', config)
    model = pretrain_tasks.BertPretrainingTasks(config)
    helper = Helper()

    if mode == 'train':
        max_steps = int(len(train_dataset) / batch_size * epoch)
        warmup_steps = int(max_steps * warmup_proportion)
        optimizer = optimization.get_optimizer(model, lr, warmup_steps, max_steps)
        criterion_lm = CrossEntropyLoss(ignore_index=-1, reduction='none')
        criterion_ns = CrossEntropyLoss(ignore_index=-1)

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, next_sentence_labels, label_ids = batch
            masked_lm_loss, next_sentence_loss = model(input_ids, segment_ids, input_mask)
            lm_labels = label_ids.view(-1)
            masked_lm_loss = criterion_lm(masked_lm_loss.view(-1, len(tokenizer)), lm_labels)
            masked_lm_loss = masked_lm_loss.sum()/(len(lm_labels) + 1e-5)
            next_sentence_loss = criterion_ns(next_sentence_loss.view(-1, 2), next_sentence_labels.view(-1))
            return masked_lm_loss + next_sentence_loss

        helper.training(process, model, train_dataset, optimizer, batch_size, epoch, model_path, save_dir, per_save_epoc)

        output_model_path = os.path.join(save_dir, "bert_model.pt")
        save(model.bert, output_model_path)

    elif mode == 'eval':

        assert model_path is not None or model_path is not '', '\"eval\" mode is model_path require'

        criterion_lm = NLLLoss(ignore_index=-1, reduction='none')
        criterion_ns = NLLLoss(ignore_index=-1)
        Example = namedtuple('Example', ('lm_pred', 'lm_true', 'ns_pred', 'ns_true'))
        if log_dir is not None or log_dir is not '':
            logger = get_logger('eval', log_dir, False)

        def process(batch, model, iter_bar, epoch, step):
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

            if log_dir is not None or log_dir is not '':
                lm_reports = classification_report(y_lm_trues, y_lm_preds, output_dict=True)
                # omit tokens score
                for k, v in lm_reports.get('micro avg').items():
                    logger.info(str('macro avg') + "," + str(k) + "," + str(v))
                for k, v in lm_reports.get('macro avg').items():
                    logger.info(str('macro avg') + "," + str(k) + "," + str(v))
                for k, v in lm_reports.get('weighted avg').items():
                    logger.info(str('weighted avg') + "," + str(k) + "," + str(v))

                ns_reports = classification_report(y_ns_trues, y_ns_preds, output_dict=True)
                for k, v in ns_reports.items():
                    for ck, cv in v.items():
                        logger.info(str(k) + "," + str(ck) + "," + str(cv))
            else:
                print(classification_report(y_ns_trues, y_ns_preds))

        helper.evaluate(process, model, train_dataset, batch_size, epoch, model_path, example_reports)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BERT pre-training.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--dataset_path', help='Dataset file path for BERT to pre-training.', nargs='?',
                        type=str, default='tests/sample_text.txt')
    parser.add_argument('--model_path', help='Pre-training PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.', nargs='?',
                        type=str, default='tests/sample_text.vocab')
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default='tests/sample_text.model')
    parser.add_argument('--save_dir', help='BERT pre-training model saving directory path.', nargs='?',
                        type=str, default='pretrain/')
    parser.add_argument('--log_dir', help='Logging file path.', nargs='?',
                        type = str, default='logs/')
    parser.add_argument('--batch_size',  help='Batch size', nargs='?',
                        type=int, default=2)
    parser.add_argument('--max_pos', help='The maximum sequence length for BERT (slow as big).', nargs='?',
                        type=int, default=128)
    parser.add_argument('--lr', help='Learning rate', nargs='?',
                        type=float, default = 5e-5)
    parser.add_argument('--warmup_steps', help='Warm-up steps proportion.', nargs='?',
                        type=float, default=0.1)
    parser.add_argument('--epoch', help='Epoch', nargs='?',
                        type=int, default=20)
    parser.add_argument('--per_save_epoc', help=
                        'Saving training model timing is the number divided by the epoch number', nargs='?',
                        type=int, default=1)
    parser.add_argument('--mode', help='train or eval', nargs='?',
                        type=str, default="train")
    args = parser.parse_args()
    bert_pretraining(args.config_path, args.dataset_path, args.model_path, args.vocab_path, args.sp_model_path,
                     args.save_dir, args.log_dir, args.batch_size, args.max_pos, args.lr, args.warmup_steps,
                     args.epoch, args.per_save_epoc, args.mode)

