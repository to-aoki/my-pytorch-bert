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


from mptb.models.bert import Config
from mptb.models.pretrain_tasks import BertPretrainingTasks, OnlyMaskedLMTasks
from mptb.models.embed_projection_albert import ProjectionOnlyMaskedLMTasks, ProjectionAlbertPretrainingTasks
from mptb.models.albert import AlbertOnlyMaskedLMTasks, AlbertPretrainingTasks
from mptb.utils import load, save


def extract_model(
    config_path='config/bert_base.json',
    model_path="pretrain/pretran_on_the_way.pt",
    output_path="pretrain/bert_only_model.pt",
    load_strict=True,
    only_bert=False,
    mlm=False,
    parallel=False,
    model_name='bert',
    pad_idx=0
):

    config = Config.from_json(config_path)
    if mlm and model_name == 'proj':
        model = ProjectionOnlyMaskedLMTasks(config, pad_idx=pad_idx)
    elif model_name == 'proj':
        model = ProjectionAlbertPretrainingTasks(config, pad_idx=pad_idx)
    elif mlm and model_name == 'albert':
        model = AlbertOnlyMaskedLMTasks(config, pad_idx=pad_idx)
    elif model_name == 'albert':
        model = AlbertPretrainingTasks(config, pad_idx=pad_idx)
    elif mlm:
        model = OnlyMaskedLMTasks(config, pad_idx=pad_idx)
    else:
        model = BertPretrainingTasks(config, pad_idx=pad_idx)

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
    parser.add_argument('--model_name', nargs='?', type=str, default='bert',
                        help=
                        'Select from the following name groups model. (bert, proj, albert)'
                        )
    parser.add_argument('--pad_idx',  help='[PAD] vocab index', nargs='?',
                        type=int, default=0)
    args = parser.parse_args()
    extract_model(config_path=args.config_path, model_path=args.model_path,
                  load_strict=not args.loose,
                  output_path=args.output_path, only_bert=args.only_bert,
                  parallel=args.parallel, mlm=args.mlm, model_name=args.model_name, pad_idx=args.pad_idx)
