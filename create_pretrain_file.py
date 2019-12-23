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
"""for Bert pre-training input feature file"""

from mptb.dataset.pretrain_dataset import PretrainDataGeneration

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BERT pre-training.', usage='%(prog)s [options]')
    parser.add_argument('--dataset_path', help='Dataset file path for BERT to pre-training.', nargs='?',
                        type=str, default='tests/sample_text.txt')
    parser.add_argument('--pickle_path', help='Pre-tensor input ids file path for BERT to pre-training.',
                        nargs='?', type=str, default=None)
    parser.add_argument('--output_path', help='Output prefix path.', nargs='?',
                        type=str, default='data/pretrain_data')
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.',
                        nargs='?', required=True, type=str)
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--max_pos', help='The maximum sequence length for BERT (slow as big).', nargs='?',
                        type=int, default=512)
    parser.add_argument('--epochs', help='Epochs', nargs='?',
                        type=int, default=20)
    parser.add_argument('--tokenizer', nargs='?', type=str, default='sp_pos',
                        help=
                        'Select from the following name groups tokenizer that uses only vocabulary files.(mecab, juman)'
                        )
    parser.add_argument('--task', nargs='?', type=str, default='mlm',
                        help='Select from the following name groups pretrain task(bert or mlm)')
    parser.add_argument('--stack', action='store_true',
                        help='Sentencestack option when task=mlm effective.')
    parser.add_argument('--onesegment_tensor', action='store_true', help='Onesegment text tensor dump pickle.')
    args = parser.parse_args()

    generator = PretrainDataGeneration(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        vocab_path=args.vocab_path,
        sp_model_path=args.sp_model_path,
        max_pos=args.max_pos,
        epochs=args.epochs,
        tokenizer_name=args.tokenizer,
        task=args.task,
        sentence_stack=args.stack,
        pickle_path=args.pickle_path
    )
    if args.onesegment_tensor:
        generator.generate_text_tensor()
    else:
        generator.generate()
