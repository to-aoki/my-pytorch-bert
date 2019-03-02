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

from models import Config
from pretrain_tasks import BertPretrainingTasks
from utils import load, save, get_device


def main(
    pretrain_model_path="pretrain/oops_google_colab_session_timeout.pt",
    bert_mode_path="collections//bert_only_model.pt",
    pyt_model_cfg='config/bert_base.json'
):
    model_cfg = Config.from_json(pyt_model_cfg)
    model = BertPretrainingTasks(model_cfg)
    load(model, pretrain_model_path, get_device())
    save(model.bert, bert_mode_path)

main()