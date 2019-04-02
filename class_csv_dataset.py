# Author Toshihiko Aoki
#
# Copyright 2018 The Google AI Language Team Authors.
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
"""CsvDataset for BERT."""

import csv
import itertools
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import truncate_seq_pair
from collections import defaultdict


class BertCsvDataset(Dataset):

    def __init__(self, tokenizer, max_pos, label_num,
                 file_path=None, sentence_a=[],sentence_b=[], labels=[],
                 delimiter='\t', encoding='utf-8', header_skip=True, under_sampling=False, cash_text=False):
        super().__init__()
        unique_labels = []
        self.records = []
        self.text_records = []

        if len(sentence_a) != 0 and len(labels) != 0 and len(sentence_a) == len(labels):
            unique_labels = list(set(labels))
            if len(sentence_b) != len(sentence_a):
                sentence_b = [None] * list(sentence_a)
            for line_a, line_b, label in sentence_a, sentence_b, labels:
                self.records.append(
                    BertCsvDataset.sentence_to_bert_ids(tokenizer, line_a, line_b, label, max_pos))

        else:
            start = 1 if header_skip else 0
            with open(file_path, "r", encoding=encoding) as f:
                csv_reader = csv.reader(f, delimiter=delimiter, quotechar=None)
                lines = tqdm(csv_reader, desc="Loading Dataset")
                for line in itertools.islice(lines, start, None):
                    assert len(line) > 1, 'require label and one sentence'
                    label = line[0]
                    sentence_a = line[1]
                    sentence_b = line[2] if len(line) > 2 else None

                    if label not in unique_labels:
                        unique_labels.append(label)

                    self.records.append(
                        BertCsvDataset.sentence_to_bert_ids(tokenizer, sentence_a, sentence_b, label, max_pos))

            assert label_num == len(unique_labels), 'label_num mismatch'
        if len(self.records) is 0:
            raise ValueError(file_path + 'were not includes documents.')

        unique_labels.sort()
        self.per_label_records_num = [0]*len(unique_labels)
        self.per_label_records = defaultdict(list)

        label_dict = {name: i for i, name in enumerate(unique_labels)}
        for record in self.records:
            record[3] = label_dict.get(record[3])  # to id
            self.per_label_records_num[record[3]] += 1
            if under_sampling:
                self.per_label_records[record[3]].append(record)

        if under_sampling:
            import random
            self.records = []
            self.under_sample_num = min(self.per_label_records_num)
            for label_num in range(len(self.per_label_records)):
                random.shuffle(self.per_label_records[label_num])
                for sample in self.per_label_records[label_num][:self.under_sample_num]:
                    self.records.append(sample)
            self.origin_per_label_records_num = self.per_label_records_num
            self.per_label_records_num = [self.under_sample_num]*len(labels)
            self.sampling_index = 1

    @staticmethod
    def sentence_to_bert_ids(tokenizer, sentence_a, sentence_b, label, max_pos):

        tokens_a = tokenizer.tokenize(sentence_a)
        tokens_b = tokenizer.tokenize(sentence_b) if sentence_b is not None else []

        # truncate
        max_seq_len = max_pos - 3 if tokens_b else max_pos - 2
        truncate_seq_pair(tokens_a, tokens_b, max_seq_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if len(tokens_b) > 0 else []

        # Add next sentence segment
        segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)

        # tokens indexing
        input_ids = tokenizer.convert_tokens_to_ids(tokens_a + tokens_b)
        input_mask = [1] * len(input_ids)

        # zero padding
        num_zero_pad = max_pos - len(input_ids)
        input_ids.extend([0] * num_zero_pad)
        segment_ids.extend([0] * num_zero_pad)
        input_mask.extend([0] * num_zero_pad)

        return [input_ids, segment_ids, input_mask, label]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return [torch.tensor(x, dtype=torch.long) for x in self.records[index]]

    def per_label_record(self):
        return self.per_label_records_num

    def next_under_samples(self):
        if self.under_sample_num is None:
            return

        self.records = []
        current_pos = self.under_sample_num * self.sampling_index
        next_pos = self.under_sample_num * (self.sampling_index+1)
        for label_num in range(len(self.per_label_records)):
            origin_num = self.origin_per_label_records_num[label_num]

            if origin_num is self.under_sample_num:
                for sample in self.per_label_records[label_num]:
                    self.records.append(sample)
                continue

            if next_pos <= origin_num:
                for sample in self.per_label_records[label_num][current_pos-1:next_pos-1]:
                    self.records.append(sample)
                continue

            if current_pos < origin_num:
                next_num = origin_num - current_pos
                for sample in self.per_label_records[label_num][current_pos-1:current_pos-1 + next_num]:
                    self.records.append(sample)
                for sample in self.per_label_records[label_num][0: self.under_sample_num - next_num]:
                    self.records.append(sample)
                continue

            sample_mod = current_pos % origin_num
            if sample_mod == 0:
                for sample in self.per_label_records[label_num][0:self.under_sample_num]:
                    self.records.append(sample)
                continue

            if origin_num < (sample_mod - 1 + self.under_sample_num):
                add_pos = (sample_mod - 1 + self.under_sample_num) - origin_num
                for sample in self.per_label_records[label_num][sample_mod-1:origin_num]:
                    self.records.append(sample)
                for sample in self.per_label_records[label_num][0:add_pos]:
                    self.records.append(sample)
            else:
                for sample in self.per_label_records[label_num][sample_mod-1:sample_mod-1 + self.under_sample_num]:
                    self.records.append(sample)

        self.sampling_index += 1
