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

import torch
from mptb.models.bert import BertModel, Config
from mptb.utils import load_from_google_bert_model


def load_tf_bert(
    config_path='config/bert_config.json',
    tfmodel_path="multi_cased_L-12_H-768_A-12/bert_model.ckpt",
    output_path="pretrain/multi_caused_L-12_H-768_A-12.pt"
):
    config = Config.from_json(config_path)
    model = BertModel(config)
    load_from_google_bert_model(model, tfmodel_path)
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert from TensorFlow BERT to my-pytorch-bert model.',
                                     usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines BERT.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--tfmodel_path', help='TensorFlow model path.', required=True,
                        type=str)
    parser.add_argument('--output_path', help='Output model path.', required=True,
                        type=str)
    args = parser.parse_args()
    load_tf_bert(args.config_path, args.tfmodel_path, args.output_path)
