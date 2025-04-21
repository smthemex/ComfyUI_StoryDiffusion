# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import diffusers
import torch

class AutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """
    def __init__(self, attention=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    def load_state_dict(self, state_dict, strict=True):
        # Newer version of diffusers changed the model keys, causing incompatibility with old checkpoints.
        # They provided a method for conversion. We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(self, "_convert_deprecated_attention_blocks", None)
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict)
