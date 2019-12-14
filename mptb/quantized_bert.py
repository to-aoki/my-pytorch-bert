# Author Toshihiko Aoki
#
# This file is based on
# https://github.com/NervanaSystems/nlp-architect/blob/master/nlp_architect/models/transformers/quantized_bert.py .
# modify for my model.
#
# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
# ******************************************************************************
# pylint: disable=bad-super-call
"""
Quantized BERT layers and model
"""

from .bert import *
from .finetuning import *
from .models import *
from nlp_architect.nn.torch.quantization import (QuantizedEmbedding, QuantizedLinear)
from nlp_architect.models.transformers.quantized_bert import quantized_linear_setup, quantized_embedding_setup


class QuantizedBertEmbeddings(Embeddings):
    def __init__(self, config, eps=1e-12):
        super(Embeddings, self).__init__()
        self.word_embeddings = quantized_embedding_setup(
            config, 'word_embeddings', config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = quantized_embedding_setup(
            config, 'position_embeddings', config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = quantized_embedding_setup(
                config, 'token_type_embeddings', config.type_vocab_size, config.hidden_size)

        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertSelfAttention(SelfAttention):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = quantized_linear_setup(
            config, 'attention_query', config.hidden_size, self.all_head_size)
        self.key = quantized_linear_setup(
            config, 'attention_key', config.hidden_size, self.all_head_size)
        self.value = quantized_linear_setup(
            config, 'attention_value', config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_data = {}  # for bertviz
        self.enable_monitor = False


class QuantizedBertSelfOutput(SelfOutput):
    def __init__(self, config, eps=1e-12):
        super(SelfOutput, self).__init__()
        self.dense = quantized_linear_setup(
            config, 'attention_output', config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertAttention(Attention):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self_attention = QuantizedBertSelfAttention(config)
        self.output = QuantizedBertSelfOutput(config)

    def prune_heads(self, heads):
        raise NotImplementedError("pruning heads is not implemented for Quantized BERT")


class QuantizedPositionwiseFeedForward(PositionwiseFeedForward):
    def __init__(self, config, eps=1e-12):
        super(PositionwiseFeedForward, self).__init__()
        self.intermediate = quantized_linear_setup(
            config, "ffn_intermediate", config.hidden_size, config.intermediate_size)
        self.output = quantized_linear_setup(
            config, "ffn_output", config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)


class QuantizedTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = QuantizedBertAttention(config)
        self.pwff = QuantizedPositionwiseFeedForward(config)


class QuantizedBertEncoder(Encoder):
    def __init__(self, config):
        super(Encoder, self).__init__()
        layer = QuantizedTransformerBlock(config)
        self.blocks_layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.attn_data_list = []
        self.attn_monitor = False


class QuantizedBertModel(BertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer")."""

    def __init__(self, config):
        super(BertModel, self).__init__()
        self.initializer_range = config.initializer_range
        self.embeddings = QuantizedBertEmbeddings(config)
        self.encoder = QuantizedBertEncoder(config)
        self.pool = quantized_linear_setup(
            config, "pooler", config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, QuantizedLinear, QuantizedEmbedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class QuantizedBertForSequenceClassification(Classification):

    def __init__(self, config, num_labels):
        super(Classification, self).__init__()
        # we only want BertForQuestionAnswering init to run to avoid unnecessary
        # initializations
        self.bert = QuantizedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = quantized_linear_setup(
            config, "head", config.hidden_size, num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias = nn.Parameter(torch.zeros(num_labels))
        self.label_len = num_labels
