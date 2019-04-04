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
"""Convert TensorFlow Bert Model to my-torch-bert."""

import tensorflow as tf
import torch
import numpy as np
from models import Config, BertModel


def copy_tf_tensor(checkpoint_dir, tf_name, pyt_attr):
    tf_tensor = tf.train.load_variable(checkpoint_dir, tf_name)
    if tf_name.endswith('kernel'):
        tf_tensor = np.transpose(tf_tensor)

    assert pyt_attr.size() == tf_tensor.shape, 'Dimension mismatch : %s:%s to %s'\
        % (tf_name, tf_tensor.shape, tuple(pyt_attr.size()))

    pyt_attr.data = torch.from_numpy(tf_tensor)


def load_from_google_bert_model(model: BertModel, f):

    s = "bert/embeddings/"
    d = model.embeddings
    copy_tf_tensor(f, s+"word_embeddings",       d.word_embeddings.weight)
    copy_tf_tensor(f, s+"position_embeddings",   d.position_embeddings.weight)
    copy_tf_tensor(f, s+"token_type_embeddings", d.token_type_embeddings.weight)
    copy_tf_tensor(f, s+"LayerNorm/gamma",       d.layer_norm.weight)
    copy_tf_tensor(f, s+"LayerNorm/beta",        d.layer_norm.bias)

    s = "bert/encoder/layer_"
    d = model.encoder.blocks_layer
    for i in range(len(model.encoder.blocks_layer)):
        copy_tf_tensor(f, s+str(i)+"/attention/self/query/kernel",      d[i].attention.self_attention.query.weight)
        copy_tf_tensor(f, s+str(i)+"/attention/self/query/bias",        d[i].attention.self_attention.query.bias)
        copy_tf_tensor(f, s+str(i)+"/attention/self/key/kernel",        d[i].attention.self_attention.key.weight)
        copy_tf_tensor(f, s+str(i)+"/attention/self/key/bias",          d[i].attention.self_attention.key.bias)
        copy_tf_tensor(f, s+str(i)+"/attention/self/value/kernel",      d[i].attention.self_attention.value.weight)
        copy_tf_tensor(f, s+str(i)+"/attention/self/value/bias",        d[i].attention.self_attention.value.bias)
        copy_tf_tensor(f, s+str(i)+"/attention/output/dense/kernel",    d[i].attention.output.dense.weight)
        copy_tf_tensor(f, s+str(i)+"/attention/output/dense/bias",      d[i].attention.output.dense.bias)
        copy_tf_tensor(f, s+str(i)+"/attention/output/LayerNorm/gamma", d[i].attention.output.layer_norm.weight)
        copy_tf_tensor(f, s+str(i)+"/attention/output/LayerNorm/beta",  d[i].attention.output.layer_norm.bias)
        copy_tf_tensor(f, s+str(i)+"/intermediate/dense/kernel",        d[i].pwff.intermediate.weight)
        copy_tf_tensor(f, s+str(i)+"/intermediate/dense/bias",          d[i].pwff.intermediate.bias)
        copy_tf_tensor(f, s+str(i)+"/output/dense/kernel",              d[i].pwff.output.weight)
        copy_tf_tensor(f, s+str(i)+"/output/dense/bias",                d[i].pwff.output.bias)
        copy_tf_tensor(f, s+str(i)+"/output/LayerNorm/gamma",           d[i].pwff.layer_norm.weight)
        copy_tf_tensor(f, s+str(i)+"/output/LayerNorm/beta",            d[i].pwff.layer_norm.bias)

    s = 'bert/pooler/'
    d = model.pool
    copy_tf_tensor(f, s+"dense/kernel", d.weight)
    copy_tf_tensor(f, s+"dense/bias",   d.bias)


def load_tf_bert(
    config_path='config/bert_base.json',
    tfmodel_path="multi_cased_L-12_H-768_A-12/bert_model.ckpt",
    vocab_num=119547,
    output_path="multi_caused_L-12_H-768_A-12.pt"
):
    config = Config.from_json(config_path, vocab_num)
    model = BertModel(config)
    load_from_google_bert_model(model, tfmodel_path)
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert from TensorFlow BERT to my-pytorch-bert model.',
                                     usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--tfmodel_path', help='TensorFlow model path.', required=True,
                        type=str)
    parser.add_argument('--vocab_num', help='Vocabulary numbers.', required=True,
                        type=int)
    parser.add_argument('--output_path', help='Output model path.', required=True,
                        type=str)
    args = parser.parse_args()
    load_tf_bert(args.config_path, args.tfmodel_path, args.vocab_num, args.output_path)
