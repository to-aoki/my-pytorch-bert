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


class BertCsvDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_pos, label_num, delimiter='\t', encoding='utf-8', header_skip=True):
        super().__init__()
        labels = []
        self.records = []
        start = 1 if header_skip else 0
        with open(file_path, "r", encoding=encoding) as f:
            csv_reader = csv.reader(f, delimiter=delimiter, quotechar=None)
            lines = tqdm(csv_reader, desc="Loading Dataset")
            for line in itertools.islice(lines, start, None):
                assert len(line) > 1, 'require label and one sentence'
                label = line[0]
                if label not in labels:
                    labels.append(label)

                tokens_a = tokenizer.tokenize(line[1])
                tokens_b = tokenizer.tokenize(line[2]) if len(line) > 2 else []

                # truncate
                max_seq_len = max_pos - 3 if tokens_b else max_pos - 2
                truncate_seq_pair(tokens_a, tokens_b, max_seq_len)

                # Add Special Tokens
                tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
                tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

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

                self.records.append([input_ids, segment_ids, input_mask, label])

        if len(self.records) is 0:
            raise ValueError(file_path + 'were not includes documents.')

        assert label_num == len(labels), 'label_num mismatch'
        labels.sort()
        for record in self.records:
            label_dict = {name: i for i, name in enumerate(labels)}
            record[3] = label_dict.get(record[3])  # to id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return [torch.tensor(x, dtype=torch.long) for x in self.records[index]]
