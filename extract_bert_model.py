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
"""Extract Bert Model."""


from mptb import Config
from mptb.pretrain_tasks import BertPretrainingTasks, OnlyMaskedLMTasks
from mptb.utils import load, save


def extract_model(
    config_path='config/bert_base.json',
    model_path="pretrain/pretran_on_the_way.pt",
    output_path="pretrain/bert_only_model.pt",
    load_strict=True,
    only_bert=False,
    mlm=False,
    parallel=False,
    albert=False,
):
    config = Config.from_json(config_path)
    if mlm:
        model = OnlyMaskedLMTasks(config, is_albert=albert)
    else:
        model = BertPretrainingTasks(config)
    if parallel:
        import torch
        model = torch.nn.DataParallel(model)
    load(model, model_path, 'cpu', strict=load_strict)
    if parallel:
        model = model.module
    if only_bert:
        model = model.bert
    save(model, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract my-pytorch-bert model.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines networks.', nargs='?',
                        type=str, default='config/bert_base.json')
    parser.add_argument('--model_path', help='my-pytorch-bert model path (include optimizer).', required=True,
                        type=str)
    parser.add_argument('--loose', action='store_true',
                        help='model load param checking loose')
    parser.add_argument('--mlm', action='store_true',
                        help='load mlm only model.')
    parser.add_argument('--parallel', action='store_true',
                        help='load parallel wrapper model.')
    parser.add_argument('--only_bert', action='store_true',
                        help='Use bert only output.')
    parser.add_argument('--output_path', help='Output model path.', required=True,
                        type=str)
    parser.add_argument('--albert', action='store_true', help='Use ALBERT model')
    args = parser.parse_args()
    extract_model(config_path=args.config_path, model_path=args.model_path,
                  load_strict=not args.loose,
                  output_path=args.output_path, only_bert=args.only_bert,
                  parallel=args.parallel, mlm=args.mlm, albert=args.albert)
