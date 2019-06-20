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

from mptb import PretrainDataGeneration


if __name__ == '__main__':
    generator = PretrainDataGeneration(
        dataset_path='tests/sample_text.txt',
        output_path='tests/sample_text',
        vocab_path='data/32023.vocab',
        sp_model_path='data/32023.model',
        max_pos=512,
        epochs=10,
        tokenizer_name='sp_pos'
    )
    generator.generate()
