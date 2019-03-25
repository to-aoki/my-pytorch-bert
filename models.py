# This file is based on
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py.
# changing class names and variables names for my understanding of BERT.
# and Modified a bit to visualize with vertviz.
#
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch BERT model."""

import copy
import math
import json

from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(NamedTuple):
    """ Configuration"""
    vocab_size: int = None                     # Vocabulary size of `inputs_ids` in `BertModel`.
    hidden_size: int = 768                     # Size of the encoder layers and the pooler layer.
    num_hidden_layers: int = 12                # Number of hidden layers in the Transformer encoder.
    num_attention_heads: int = 12              # Number of attention heads for each attention layer.
    intermediate_size: int = 768*4             # The size of the "intermediate" layer in the Transformer encoder.
    hidden_dropout_prob: float = 0.1           # The dropout probability for hidden layers.
    attention_probs_dropout_prob: float = 0.1  # The dropout ratio for the attention probabilities.
    max_position_embeddings: int = 128         # The maximum sequence length (slow as big).
    type_vocab_size: int = 2                   # The vocabulary size of the `token_type_ids` passed into `BertModel`.
    initializer_range: float = 0.02            # initialize weight range

    @classmethod
    def from_json(cls, file, vocab_size=None, max_position_embeddings=None, type_vocab_size=None):
        config = json.load(open(file, "r"))
        if vocab_size is not None:
            config['vocab_size'] = vocab_size
        if max_position_embeddings is not None:
            config['max_position_embeddings'] = max_position_embeddings
        if type_vocab_size is not None:
            config['type_vocab_size'] = type_vocab_size
        return cls(**config)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:

    class LayerNorm(nn.Module):
        """A layernorm module in the TF style (epsilon inside the square root)."""
        def __init__(self, hidden_size, eps=1e-12):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))   # gamma
            self.bias = nn.Parameter(torch.zeros(hidden_size))    # beta
            self.variance_epsilon = eps

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean)**2).mean(dim=-1, keepdim=True)
            std = (var + self.variance_epsilon).sqrt()
            return self.weight * (x - mean)/std + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        max_position_embeddings = input_ids.size(1)
        position_ids = torch.arange(max_position_embeddings,  dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = self.word_embeddings(input_ids) \
            + self.position_embeddings(position_ids) \
            + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.layer_norm(embeddings))


class SelfAttention(nn.Module):
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def forward(self, hidden_states, attention_mask, monitor=False):
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

        context_layer = context_layer.transpose(2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # for bertviz
        if monitor:
            self.attn_data = {
                'attn_probs': attention_probs,
                'query_layer': query_layer,
                'key_layer': key_layer
            }

        return context_layer
    
    # for vertviz
    def monitor(self):
        return self.attn_data


class SelfOutput(nn.Module):
    def __init__(self, config, eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask, monitor=False):
        self_attention_output = self.self_attention(input_tensor, attention_mask, monitor)
        attention_output = self.output(self_attention_output, input_tensor)
        return attention_output


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, config, eps=1e-12):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)

    def forward(self, attention_output):
        intermediate_output = gelu(self.intermediate(attention_output))
        hidden_states = self.dropout(self.output(intermediate_output))
        return self.layer_norm(hidden_states + attention_output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, hidden_states, attention_mask, monitor=False):
        attention_output = self.attention(hidden_states, attention_mask, monitor)
        return self.pwff(attention_output)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = TransformerBlock(config)
        self.blocks_layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.attn_data_list = []

    def forward(self, hidden_states, attention_mask, monitor=False):
        self.attn_data_list = []
        for layer_module in self.blocks_layer:
            hidden_states = layer_module(hidden_states, attention_mask, monitor)
            if monitor:
                self.attn_data_list.append(layer_module.attention.self_attention.monitor())
        return hidden_states
    
    # for vertviz
    def monitor(self):
        return self.attn_data_list


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer")."""
    def __init__(self, config):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.pool = nn.Linear(config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # tiny truncate norm
            module.weight.data = torch.fmod(
                torch.randn(module.weight.size()), self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, monitor=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        hidden_states = self.encoder(embedding_output, extended_attention_mask, monitor)
        pooled_output = torch.tanh(self.pool(hidden_states[:, 0]))

        return hidden_states, pooled_output

