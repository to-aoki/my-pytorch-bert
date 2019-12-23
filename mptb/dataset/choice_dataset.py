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
"""Choice Dataset for BERT."""

import csv
import itertools
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from mptb.utils import to_bert_ids


class SwagDataset(Dataset):

    def __init__(
        self,
        tokenizer, max_pos, dataset_path=None, delimiter=',', encoding='utf-8', header_skip=True, is_training=True
    ):
        super().__init__()
        self.records = []

        start = 1 if header_skip else 0
        with open(dataset_path, "r", encoding=encoding) as f:
            csv_reader = csv.reader(f, delimiter=delimiter)
            lines = tqdm(csv_reader, desc="Loading Dataset")
            for line in itertools.islice(lines, start, None):
                swag_record = [[], [], []]
                # swag_id = line[2],
                context_sentence = line[4]
                start_ending = line[5]
                label = int(line[11]) if is_training else None   # False is not test
                for ending in line[7:11]:
                    bert_ids = to_bert_ids(max_pos, tokenizer, context_sentence, start_ending + ending)
                    swag_record[0].append(bert_ids[0])  # input_ids
                    swag_record[1].append(bert_ids[1])  # segment_type
                    swag_record[2].append(bert_ids[2])  # input_masks
                if is_training:
                    swag_record.append(label)
                self.records.append(swag_record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return [torch.tensor(x, dtype=torch.long) for x in self.records[index]]

