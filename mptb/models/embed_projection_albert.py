# This file is based on
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py.
# Extracted common classes of Bert.
#
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embed Projection ALBERT model."""

import torch
import torch.nn as nn
from .commons import LayerNorm, gelu
from .bert import BertModel, TransformerBlock
from .pretrain_tasks import NextSentencePrediction


class ProjectionEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.projection = nn.Linear(config.embedding_size, config.hidden_size, bias=False)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            max_position_embeddings = input_ids.size(1)
            position_ids = torch.arange(max_position_embeddings, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = self.word_embeddings(input_ids)
        embeddings = self.projection(embeddings)
        if hasattr(self, 'token_type_embeddings'):
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            embeddings += self.token_type_embeddings(token_type_ids)
        embeddings += self.position_embeddings(position_ids)

        return self.dropout(self.layer_norm(embeddings))


class OneLayerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_hidden_layers = config.num_hidden_layers
        layer = TransformerBlock(config)
        self.blocks_layer = layer
        self.attn_data_list = []
        self.attn_monitor = False

    def enable_monitor(self):
        self.attn_monitor = True
        self.blocks_layer.attention.self_attention.enable_monitor = True
        return

    def forward(self, hidden_states, attention_mask, layer=-1):
        if self.attn_monitor:
            self.attn_data_list = []
        all_hidden_states = []
        for i in range(self.num_hidden_layers):
            layer_module = self.blocks_layer
            hidden_states = layer_module(hidden_states, attention_mask)
            if self.attn_monitor:
                self.attn_data_list.append(layer_module.attention.self_attention.attn_data)
            all_hidden_states.append(hidden_states)
        return all_hidden_states[layer]

    # for bertviz
    def monitor(self):
        return self.attn_data_list


class EmbedProjectionAlbertModel(BertModel):
    """ A Lite Bert For Self-Supervised Learning Language Representations."""

    def __init__(self, config):
        super(EmbedProjectionAlbertModel, self).__init__(config)
        self.embeddings = ProjectionEmbeddings(config)
        self.encoder = OneLayerEncoder(config)
        self.apply(self.init_bert_weights)


class ProjectionOnlyMaskedLMTasks(nn.Module):
    """Pre-training Tasks only MaskedLM"""

    def __init__(self, config):
        super().__init__()
        self.bert = EmbedProjectionAlbertModel(config)
        self.masked_lm = ProjectionMaskedLM(config, self.bert.embeddings.word_embeddings.weight.size(0))

    def forward(self, input_ids, segment_ids, input_mask):
        hidden_state, _ = self.bert(input_ids, segment_ids, input_mask)
        logits_lm = self.masked_lm.forward(
            hidden_state, self.bert.embeddings.word_embeddings.weight, self.bert.embeddings.projection.weight)
        return logits_lm, None


class ProjectionAlbertPretrainingTasks(nn.Module):
    """Alert Pre-training Tasks"""

    def __init__(self, config):
        super().__init__()
        self.bert = EmbedProjectionAlbertModel(config)
        self.masked_lm = ProjectionMaskedLM(config, self.bert.embeddings.word_embeddings.weight.size(0))
        self.next_sentence_prediction = NextSentencePrediction(config)

    def forward(self, input_ids, segment_ids, input_mask):
        hidden_state, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        logits_lm = self.masked_lm.forward(
            hidden_state, self.bert.embeddings.word_embeddings.weight, self.bert.embeddings.projection.weight)
        logits_nsp = self.next_sentence_prediction(pooled_output)
        return logits_lm, logits_nsp


class ProjectionMaskedLM(nn.Module):
    """Pre-training #1 : Masked LM"""

    def __init__(self, config, n_vocab):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.bias = nn.Parameter(torch.zeros(n_vocab))
        self.dense.weight.data = torch.fmod(
            torch.randn(self.dense.weight.size()), config.initializer_range)

    def forward(self, hidden_states, word_embeddings_weight, projection_weight=None):
        hidden_states = gelu(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        if projection_weight is not None:
            hidden_states = torch.matmul(hidden_states, projection_weight)
        return torch.matmul(hidden_states, word_embeddings_weight.transpose(0, 1)) + self.bias
