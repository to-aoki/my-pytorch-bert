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

"""PyTorch ALBERT model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import LayerNorm
from .bert import BertModel, Attention, TransformerBlock
from .pretrain_tasks import NextSentencePrediction, OnlyMaskedLMTasks, BertPretrainingTasks


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)

        self.layer_norm = LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            max_position_embeddings = input_ids.size(1)
            position_ids = torch.arange(max_position_embeddings, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = self.word_embeddings(input_ids)
        if hasattr(self, 'token_type_embeddings'):
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            embeddings += self.token_type_embeddings(token_type_ids)
        embeddings += self.position_embeddings(position_ids)

        return self.dropout(self.layer_norm(embeddings))


class AlbertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_data = {}  # for bertviz
        self.enable_monitor = False

        self.hidden_size = config.hidden_size
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def forward(self, hidden_states, attention_mask):
        self.attn_data = {}  # for bertviz
        mixed_query_layer = self.query(hidden_states).contiguous()
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot proattn_dataduct between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        w = self.projection.weight.t().view(
            self.num_attention_heads, self.attention_head_size, self.hidden_size).to(context_layer.dtype)
        b = self.projection.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        context_layer = self.layer_norm(hidden_states + projected_context_layer_dropout)

        # for bertviz
        if self.enable_monitor:
            self.attn_data = {
                'attn_probs': attention_probs,
                'query_layer': query_layer,
                'key_layer': key_layer
            }

        return context_layer


class AlbertAttention(Attention):
    def __init__(self, config):
        super(AlbertAttention, self).__init__(config)
        self.self_attention = AlbertSelfAttention(config)


class AlbertTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super(AlbertTransformerBlock, self).__init__(config)
        self.attention = AlbertAttention(config)


class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        layer = AlbertTransformerBlock(config)
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.blocks_layer = layer
        self.attn_data_list = []
        self.attn_monitor = False

    def enable_monitor(self):
        self.attn_monitor = True
        self.blocks_layer.attention.self_attention.enable_monitor = True

    def forward(self, hidden_states, attention_mask, layer=-1):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

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


class AlbertModel(BertModel):
    """ A Lite Bert For Self-Supervised Learning Language Representations."""

    def __init__(self, config):
        super(AlbertModel, self).__init__(config)
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.apply(self.init_bert_weights)


class AlbertOnlyMaskedLMTasks(OnlyMaskedLMTasks):
    """Bert Pre-training Tasks"""

    def __init__(self, config):
        super(AlbertOnlyMaskedLMTasks, self).__init__(config)
        self.bert = AlbertModel(config)
        self.masked_lm = AlbertMaskedLM(config, self.bert.embeddings.word_embeddings.weight.size(0))


class AlbertPretrainingTasks(BertPretrainingTasks):
    """Bert Pre-training Tasks"""

    def __init__(self, config):
        super(AlbertPretrainingTasks, self).__init__(config)
        self.bert = AlbertModel(config)
        self.masked_lm = AlbertMaskedLM(config, self.bert.embeddings.word_embeddings.weight.size(0))
        self.next_sentence_prediction = NextSentencePrediction(config)


class AlbertMaskedLM(nn.Module):
    """Bert Pre-training #1 : Masked LM"""

    def __init__(self, config, n_vocab):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_vocab))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.dense.weight.data = torch.fmod(
            torch.randn(self.dense.weight.size()), config.initializer_range)
        self.decoder = nn.Linear(config.embedding_size, n_vocab)
        self.layer_norm = LayerNorm(config.embedding_size, eps=1e-12)

    def forward(self, hidden_states, word_embeddings_weight, projection_weight=None):
        hidden_states = gelu(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states + self.bias
