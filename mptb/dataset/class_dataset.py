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
"""Class Dataset for BERT."""

import csv
import itertools
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from mptb.utils import to_bert_ids
from collections import defaultdict


class ClassDataset(Dataset):

    def __init__(
        self,
        tokenizer, max_pos, label_num=-1,
        dataset_path=None, delimiter='\t', encoding='utf-8', header_skip=True,
        sentence_a=[], sentence_b=[], labels=[],
        under_sampling=False
    ):
        super().__init__()
        unique_labels = []
        self.records = []
        self.text_records = []
        self.per_label_records_num = []
        self.max_pos = max_pos

        bert_unk_id = tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
        sp_unk_id = tokenizer.convert_tokens_to_ids(["<unk>"])[0]
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        if pad_id == bert_unk_id:
            if bert_unk_id != sp_unk_id:
                import warnings
                warnings.warn('<unk> included.')
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'

        if len(sentence_a) > 0:
            if len(sentence_b) != len(sentence_a):
                sentence_b = [None] * list(sentence_a)
            if 0 < len(labels) == len(sentence_a):
                unique_labels = list(set(labels))
                for line_a, line_b, label in sentence_a, sentence_b, labels:
                    bert_ids = to_bert_ids(max_pos, tokenizer, line_a, line_b, pad_token=pad_token)
                    bert_ids.append(label)
                    self.records.append(bert_ids)
            else:
                for line_a, line_b in sentence_a, sentence_b:
                    self.records.append(
                        to_bert_ids(max_pos, tokenizer, line_a, line_b))

        else:
            start = 1 if header_skip else 0
            with open(dataset_path, "r", encoding=encoding) as f:
                csv_reader = csv.reader(f, delimiter=delimiter, quotechar=None)
                lines = tqdm(csv_reader, desc="Loading Dataset")
                for line in itertools.islice(lines, start, None):
                    assert len(line) > 1, 'require label and one sentence : ' + str(line)
                    label = line[0]
                    sentence_a = line[1]
                    sentence_b = line[2] if len(line) > 2 else None

                    if label not in unique_labels:
                        unique_labels.append(label)
                    bert_ids = to_bert_ids(max_pos, tokenizer, sentence_a, sentence_b)
                    bert_ids.append(label)
                    self.records.append(bert_ids)

            if label_num > 0:
                assert label_num == len(unique_labels), 'label_num mismatch'
        if len(self.records) is 0:
            raise ValueError(dataset_path + 'were not includes documents.')

        if unique_labels:
            unique_labels.sort()
            self.per_label_records_num = [0]*len(unique_labels)
            self.per_label_records = defaultdict(list)

            label_dict = {name: i for i, name in enumerate(unique_labels)}
            for record in self.records:
                record[3] = label_dict.get(record[3])  # label to id
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
                self.per_label_records_num = [self.under_sample_num]*len(unique_labels)
                self.sampling_index = 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return [torch.tensor(x, dtype=torch.long) for x in self.records[index]]

    def per_label_record(self):
        return self.per_label_records_num

    def label_num(self):
        return len(self.per_label_records_num)

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
