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
    only_bert=False,
    mlm=False,
    parallel=False,
    apex=False
):
    config = Config.from_json(config_path)
    if mlm:
        model = OnlyMaskedLMTasks(config)
    else:
        model = BertPretrainingTasks(config)
    if apex:
        model.to('cuda')
        from mptb import get_optimizer
        optimizer = get_optimizer(
            model=model)
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, _ = amp.initialize(model, optimizer, opt_level='O2')
    if parallel:
        import torch
        model = torch.nn.DataParallel(model)

    load(model, model_path, 'cpu')
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
    parser.add_argument('--mlm', action='store_true',
                        help='Use mlm only model.')
    parser.add_argument('--apex', action='store_true',
                        help='apex module converted.')
    parser.add_argument('--parallel', action='store_true',
                        help='parallel module converted.')
    parser.add_argument('--only_bert', action='store_true',
                        help='Use bert only output.')
    parser.add_argument('--output_path', help='Output model path.', required=True,
                        type=str)
    args = parser.parse_args()
    extract_model(config_path=args.config_path, model_path=args.model_path,
                  output_path=args.output_path, only_bert=args.only_bert,
                  apex=args.apex, parallel=args.parallel, mlm=args.mlm)
