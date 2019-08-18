# Author Toshihiko Aoki
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Bert fine-tuning models."""

import torch
import torch.nn as nn

from . bert import BertModel


class Classifier(nn.Module):
    """Bert fine-tuning classifier"""

    def __init__(self, config, num_labels):
        super().__init__()

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.weight.data = torch.fmod(
            torch.randn(self.classifier.weight.size()), config.initializer_range)
        self.classifier.bias = nn.Parameter(torch.zeros(num_labels))
        self.label_len = num_labels

    def forward(self, input_ids, segment_ids, input_mask):
        _, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        return self.classifier(self.dropout(pooled_output))


class TokenClassifier(nn.Module):
    """Bert fine-tuning Token classifier"""

    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.weight.data = torch.fmod(
            torch.randn(self.classifier.weight.size()), config.initializer_range)
        self.classifier.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, segment_ids, input_mask):
        hidden_state, _ = self.bert(input_ids, segment_ids, input_mask)
        return self.classifier(self.dropout(hidden_state))


class MultipleChoiceSelector(nn.Module):
    """Bert fine-tuning Token classifier"""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier.weight.data = torch.fmod(
            torch.randn(self.classifier.weight.size()), config.initializer_range)
        self.classifier.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, segment_ids, input_mask, position_ids=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_segment_ids = segment_ids.view(-1, segment_ids.size(-1)) if segment_ids is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        _, pooled_output = self.bert(flat_input_ids, flat_segment_ids, flat_input_mask, flat_position_ids)

        logits = self.classifier(self.dropout(pooled_output))
        return logits.view(-1, num_choices)


class QuestionRespondent(nn.Module):
    """Bert fine-tuning Question respondent"""

    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, num_labels)
        self.qa_outputs.weight.data = torch.fmod(
            torch.randn(self.classifier.weight.size()), config.initializer_range)
        self.qa_outputs.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, segment_ids, input_mask, position_ids):
        hidden_state, _ = self.bert(input_ids, segment_ids, input_mask,  position_ids)
        logits = self.qa_outputs(hidden_state)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits
