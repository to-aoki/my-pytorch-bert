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


def main(model_cfg='config/bert_base.json',
         data_file='tests/sample_text.txt',
         model_file=None,
         vocab_file='tests/sample_text.vocab',
         sp_model_file='tests/sample_text.model',
         save_dir='pretrain/',
         log_dir='logs/',
         batch_size=2,
         max_len=128,
         lr=5e-5,
         warmup_proportion=0.1,  # warmup_steps = len(dataset) / batch_size * epoch * warmup_proportion
         epoch=5,
         per_save_epoc=1,
         mode='train'):

    assert mode is not None and (mode is 'train' or mode is 'eval'), 'support mode train or eval.'

    if sp_model_file is not None:
        tokenizer = tokenization_sentencepiece.FullTokenizer(
            sp_model_file, vocab_file, do_lower_case=True)
    else:
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    if max_len is None:
        # max_len = statistics.median(all-sentence-tokens)
        import statistics
        with open(data_file, 'r', newline="\n", encoding="utf-8") as data:
            tokens = list(map(tokenizer.tokenize, data.readlines()))
            max_len = round(statistics.median(list(map(lambda x: len(x), tokens))))   
        max_len = max_len*2+3  # [CLS]a[SEP]b[SEP]
        print("max_len (median):", max_len)

    train_dataset = PretrainDataset(
        data_file,
        tokenizer,
        max_pos=max_len,
        corpus_lines=None,
        on_memory=True
    )

    model_cfg = models.Config.from_json(model_cfg, len(tokenizer), max_len)
    print('model params :', model_cfg)
    model = pretrain_tasks.BertPretrainingTasks(model_cfg)
    helper = Helper()

    if mode is 'train':

        warmup_steps = int(len(train_dataset) / batch_size * epoch * warmup_proportion)
        optimizer = optimization.get_optimizer(model, lr, warmup_steps)
        criterion_lm = CrossEntropyLoss(ignore_index=-1, reduction='none')
        criterion_ns = CrossEntropyLoss(ignore_index=-1, reduction='none')

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, next_sentence_labels = batch
            masked_lm_loss, next_sentence_loss = model(input_ids, segment_ids, input_mask)
            lm_labels = input_ids.view(-1)
            masked_lm_loss = criterion_lm(masked_lm_loss.view(-1, len(tokenizer)), lm_labels)
            masked_lm_loss = masked_lm_loss.sum()/len(lm_labels) + 1e-5
            next_sentence_loss = criterion_ns(next_sentence_loss.view(-1, 2), next_sentence_labels.view(-1)).mean()
            return masked_lm_loss + next_sentence_loss

        helper.training(process, model, train_dataset, optimizer, batch_size, epoch, model_file, save_dir, per_save_epoc)

        output_model_file = os.path.join(save_dir, "bert_model.pt")
        save(model.bert, output_model_file)

    elif mode is 'eval':

        assert model_file is not None or model_file is not '', '\"eval\" mode is model_file require'

        criterion_lm = NLLLoss(ignore_index=-1, reduction='none')
        criterion_ns = NLLLoss(ignore_index=-1, reduction='none')
        Example = namedtuple('Example', ('lm_pred', 'lm_true', 'ns_pred', 'ns_true'))
        logger = get_logger('eval', log_dir, False)

        def process(batch, model, iter_bar, epoch, step):
            input_ids, segment_ids, input_mask, next_sentence_labels = batch
            masked_lm_loss, next_sentence_loss = model(input_ids, segment_ids, input_mask)

            masked_lm_probs = F.log_softmax(masked_lm_loss.view(-1, len(tokenizer)), 1)
            masked_lm_predictions = masked_lm_probs.max(1, False)
            lm_labels = input_ids.view(-1)

            ns_probs = F.log_softmax(next_sentence_loss.view(-1, 2), 1)
            ns_predictions = ns_probs.max(1, False)
            ns_labels = next_sentence_labels.view(-1)

            example = Example(
                masked_lm_predictions[1].tolist(), lm_labels.tolist(),
                ns_predictions[1].tolist(), ns_labels.tolist())

            masked_lm_loss = criterion_lm(masked_lm_probs, lm_labels)
            masked_lm_loss = masked_lm_loss.sum()/len(lm_labels) + 1e-5
            next_sentence_loss = criterion_ns(ns_probs, ns_labels).mean()

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

            lm_reports = classification_report(y_lm_trues, y_lm_preds, output_dict=True)
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

        helper.evaluate(process, model, train_dataset, batch_size, epoch, model_file, example_reports)


main(mode='train')
