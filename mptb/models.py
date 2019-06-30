# This file is based on
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py.
# changing class names and variables names for my understanding of BERT.
# and Modified a bit to visualize with bertviz.
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
"""Common Network model."""

import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
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


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, config, eps=1e-12):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)

    def forward(self, attention_output):
        intermediate_output = self.dropout(gelu(self.intermediate(attention_output)))
        hidden_states = self.dropout(self.output(intermediate_output))
        return self.layer_norm(hidden_states + attention_output)


