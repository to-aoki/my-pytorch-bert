# Author Toshihiko Aoki
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
"""Run Bert SWEM."""

import numpy as np
from mptb import BertSWEM


def swem(
    config_path='../config/bert_base.json',
    max_pos=-1,
    vocab_path=None,
    sp_model_path=None,
    tokenizer_name='google',
    bert_model_path=None,
    device='cpu',
    text_a='吾輩は猫である。',
    text_b=None,
    layer='-1',
    strategy='REDUCE_MEAN'
):
    swem = BertSWEM(
        config_path=config_path,
        max_pos=max_pos,
        vocab_path=vocab_path,
        sp_model_path=sp_model_path,
        tokenizer_name=tokenizer_name,
        bert_model_path=bert_model_path,
        device=device,
    )
    text_a_vector = swem.embedding_vector(text_a, pooling_layer=layer, pooling_strategy=strategy)
    print(text_a, text_a_vector)
    if text_b is not None:
        text_b_vector = swem.embedding_vector(text_b, pooling_layer=layer, pooling_strategy=strategy)
        print(text_b, text_b_vector)
        text_a_vector = np.array(text_a_vector)
        text_b_vector = np.array(text_b_vector)
        print('cosine similarity',
              np.dot(text_a_vector, text_b_vector) / (np.linalg.norm(text_a_vector) * np.linalg.norm(text_b_vector)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract my-pytorch-bert model.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--max_pos', help='The maximum sequence length for BERT (slow as big).', nargs='?',
                        type=int, default=512)
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.', nargs='?', required=True,
                        type=str)
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--tokenizer', nargs='?', type=str, default='google',
                        help=
                        'Select from the following name groups tokenizer that uses only vocabulary files.(mecab, juman)'
                        )
    parser.add_argument('--model_path', help='PyTorch BERT model path.', nargs='?', type=str, required=True,)
    parser.add_argument('--device', nargs='?', type=str, default='cpu', help='Target Runing device name.')
    parser.add_argument('--text', nargs='?', type=str, help='Sentence', required=True,)
    parser.add_argument('--compare', nargs='?', type=str, default='None', help='Compare sentence')
    parser.add_argument('--layer', nargs='?', type=int, default='-1', help='Use Bert pooling layer')
    parser.add_argument('--strategy', nargs='?', type=str, default='REDUCE_MEAN',
                        help='Use SWEM operation (REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN)')

    args = parser.parse_args()
    swem(config_path=args.config_path, max_pos=args.max_pos,
         vocab_path=args.vocab_path, sp_model_path=args.sp_model_path, tokenizer_name=args.tokenizer,
         bert_model_path=args.model_path, device=args.device,
         text_a=args.text, text_b=args.compare, layer=args.layer, strategy=args.strategy)
