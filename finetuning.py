# Author Toshihiko Aoki
#
# Copyright 2018 The Google AI Language Team Authors
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

import torch
import torch.nn as nn

from models import BertModel


class Classifier(nn.Module):
    """Bert Fine-tuning usecase classifier"""

    def __init__(self, config, label_len):
        super().__init__()

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, label_len)
        self.classifier.weight.data = torch.fmod(
            torch.randn(self.classifier.weight.size()), config.initializer_range)
        self.classifier.bias = nn.Parameter(torch.zeros(label_len))

    def forward(self, input_ids, segment_ids, input_mask):
        _, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        return self.classifier(pooled_output)
