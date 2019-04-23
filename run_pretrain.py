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
"""Bert pre-training."""

from mptb import BertPretrainEstimator


def bert_pretraining(
    config_path='config/bert_base.json',
    dataset_path='tests/sample_text.txt',
    model_path=None,
    vocab_path='tests/sample_text.vocab',
    sp_model_path='tests/sample_text.model',
    save_dir='pretrain/',
    log_dir=None,
    batch_size=2,
    max_pos=128,
    lr=5e-5,
    warmup_proportion=0.1,  # warmup_steps = len(dataset) / batch_size * epoch * warmup_proportion
    epoch=5,
    per_save_epoch=1,
    mode='train',
    use_mecab=False
):

    estimataor = BertPretrainEstimator(
        config_path=config_path,
        max_pos=max_pos,
        vocab_path=vocab_path,
        sp_model_path=sp_model_path,
        dataset_path=dataset_path,
        use_mecab=use_mecab
    )

    if mode == 'train':
        estimataor.train(
            traing_model_path=model_path, batch_size=batch_size, epoch=epoch, per_save_epoch=per_save_epoch,
            lr=lr, warmup_proportion=warmup_proportion, save_dir=save_dir
        )
        score = estimataor.evaluate(batch_size=batch_size, log_dir=log_dir)
        print(score)
    else:
        score = estimataor.evaluate(model_path=model_path, batch_size=batch_size, log_dir=log_dir)
        print(score)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BERT pre-training.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--dataset_path', help='Dataset file path for BERT to pre-training.', nargs='?',
                        type=str, default='tests/sample_text.txt')
    parser.add_argument('--model_path', help='Pre-training PyTorch model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--vocab_path', help='Vocabulary file path for BERT to pre-training.', nargs='?', required=True,
                        type=str)
    parser.add_argument('--sp_model_path', help='Trained SentencePiece model path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--save_dir', help='BERT pre-training model saving directory path.', nargs='?',
                        type=str, default='pretrain/')
    parser.add_argument('--log_dir', help='Logging file path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--batch_size',  help='Batch size', nargs='?',
                        type=int, default=4)
    parser.add_argument('--max_pos', help='The maximum sequence length for BERT (slow as big).', nargs='?',
                        type=int, default=128)
    parser.add_argument('--lr', help='Learning rate', nargs='?',
                        type=float, default = 5e-5)
    parser.add_argument('--warmup_steps', help='Warm-up steps proportion.', nargs='?',
                        type=float, default=0.1)
    parser.add_argument('--epoch', help='Epoch', nargs='?',
                        type=int, default=20)
    parser.add_argument('--per_save_epoch', help=
                        'Saving training model timing is the number divided by the epoch number', nargs='?',
                        type=int, default=1)
    parser.add_argument('--mode', help='train or eval', nargs='?',
                        type=str, default="train")
    parser.add_argument('--use_mecab', action='store_true',
                        help='Use Mecab Tokenizer')
    args = parser.parse_args()
    bert_pretraining(args.config_path, args.dataset_path, args.model_path, args.vocab_path, args.sp_model_path,
                     args.save_dir, args.log_dir, args.batch_size, args.max_pos, args.lr, args.warmup_steps,
                     args.epoch, args.per_save_epoch, args.mode, args.use_mecab)

