# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import torch.nn as nn
import importlib

def zero_module(module):
    if isinstance(module, nn.Linear):
        module.weight.data.zero_()
        if module.bias is not None:
            module.bias.data.zero_()
    return module

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def update_dict(old_dict, new_dict):
    old_keys = old_dict.keys()
    for new_key in new_dict.keys(): 
        if new_key in old_keys:
            if type(old_dict[new_key]) == list:
                if type(new_dict[new_key]) == list:
                    old_dict[new_key].extend(new_dict[new_key])
                else:
                    old_dict[new_key].append(new_dict[new_key])
            else:
                old_dict[new_key] = [old_dict[new_key]]
                old_dict[new_key].append(new_dict[new_key])
        else:
            old_dict[new_key] = new_dict[new_key]
    return old_dict