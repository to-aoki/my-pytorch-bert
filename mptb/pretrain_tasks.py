# Author Toshihiko Aoki
#
# Copyright 2018 The Google AI Language Team Authors
#
# Licsed under the Apache License, Version 2.0 (the "License");
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
"""Bert pre-training tasks."""


import torch
import torch.nn as nn

from .models import gelu, LayerNorm
from .bert import BertModel


class OnlyMaskedLMTasks(nn.Module):
    """Bert Pre-training Tasks"""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.masked_lm = MaskedLM(config, self.bert.embeddings.word_embeddings.weight.size(0))

    def forward(self, input_ids, segment_ids, input_mask):
        hidden_state, _ = self.bert(input_ids, segment_ids, input_mask)
        logits_lm = self.masked_lm.forward(hidden_state, self.bert.embeddings.word_embeddings.weight)
        return logits_lm, None


class BertPretrainingTasks(nn.Module):
    """Bert Pre-training Tasks"""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.masked_lm = MaskedLM(config, self.bert.embeddings.word_embeddings.weight.size(0))
        self.next_sentence_prediction = NextSentencePrediction(config)

    def forward(self, input_ids, segment_ids, input_mask):
        hidden_state, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        logits_lm = self.masked_lm.forward(hidden_state, self.bert.embeddings.word_embeddings.weight)
        logits_nsp = self.next_sentence_prediction(pooled_output)
        return logits_lm, logits_nsp


class MaskedLM(nn.Module):
    """Bert Pre-training #1 : Masked LM"""

    def __init__(self, config, n_vocab):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_vocab))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.weight.data = torch.fmod(
            torch.randn(self.dense.weight.size()), config.initializer_range)
        self.layer_norm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = gelu(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        return torch.matmul(hidden_states, word_embeddings_weight.transpose(0, 1)) + self.bias


class NextSentencePrediction(nn.Module):
    """Bert Pre-training #2 : Next Sentence Prediction"""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.classifier.bias = nn.Parameter(torch.zeros(2))
        self.classifier.weight.data = torch.fmod(
            torch.randn(self.classifier.weight.size()), config.initializer_range)

    def forward(self, pooled_output):
        return self.classifier(pooled_output)

