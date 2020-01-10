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
"""MeCab vocab builder."""

from mptb.tokenization.tokenization_mecab import *
from mptb.tokenization.preprocessing import *
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='for MeCab Tokenizer vocab file generate.', usage='%(prog)s [options]')
    parser.add_argument('--file_path', help='Original text file path', required=True,
                        type=str)
    parser.add_argument('--vocab_path', help='Output vocab file path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--convert_path', help='Output convert file path.', nargs='?',
                        type=str, default=None)
    parser.add_argument('--min_freq', help='Word appearance frequency adopted as vocabulary', nargs='?',
                        type=int, default=1)
    parser.add_argument('--limit_vocab_length', help='Word appearance frequency adopted as vocabulary', nargs='?',
                        type=int, default=-1)
    args = parser.parse_args()
    if args.vocab_path is not None:
        print('created : ' + args.vocab_path + ' , size :' + str(
            create_vocab(args.file_path, args.vocab_path, args.min_freq, args.limit_vocab_length)))
        sys.exit(0)

    preprocessor = Pipeline([
        ToUnicode(),
        Normalize(),
        LowerCase(),
        ReplaceURI(),
    ])

    tokenizer = MeCabTokenizer(preprocessor=preprocessor)
    if args.convert_path is not None:
        _, ext = os.path.splitext(args.file_path)
        with open(args.file_path, "r", encoding='utf-8') as reader:
            with open(args.convert_path, 'w', encoding='utf-8', newline="\n") as writer:
                if ext == '.tsv':
                    for line in reader:
                        split = line.split('\t')
                        writer.write(split[1].strip() + '\t' + ' '.join(tokenizer.tokenize(split[0])).strip() + '\n')
                else:
                    for line in reader:
                        writer.write(' '.join(tokenizer.tokenize(line)).strip() + '\n')
    else:
        with open(args.file_path, "r", encoding='utf-8') as reader:
            for line in reader:
                print(' '.join(tokenizer.tokenize(line)))

