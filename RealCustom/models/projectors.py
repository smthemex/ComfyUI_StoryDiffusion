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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(kv_dim)

        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)

    def forward(self, local_fea, global_fea):
        global_fea = self.norm1(global_fea)
        local_fea = self.norm2(local_fea)
        b, l, _ = global_fea.shape

        q = self.to_q(global_fea)
        k = self.to_k(local_fea)
        v = self.to_v(local_fea)

        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            q,k,v, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(b, -1, self.heads*self.dim_head)
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states

class CustomParameter(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.init_value = init_value
        self.value = nn.Parameter(torch.tensor(init_value))
    
    def forward(self):
        return self.value


class ProjectorHighResMinAttn(nn.Module):
    def __init__(self, vision_dim, out_dim, dim_head=64, adaptive_scale=False, scale_value=1.0, **kwargs):
        super().__init__()
        self.initial_projection_dim = vision_dim * 4
        heads = vision_dim // dim_head

        self.min_attention = MinAttention(q_dim=vision_dim, kv_dim=vision_dim, dim_head=dim_head, heads=heads)
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim)
        )
        self.projector_base = nn.Linear(vision_dim, out_dim, bias=True)

        self.adaptive_scale = adaptive_scale
        if self.adaptive_scale:
            self.scale_value = CustomParameter(scale_value)

    def forward(self, vision_input_dict, time_emb=None, **kwargs):
        """
        vision_input_dict: here, this is not a dict, just for the unity of naming
        """
        img_patch_features = vision_input_dict
        deep_features, deep_features_local = img_patch_features

        fused_img_features = self.min_attention(deep_features_local, deep_features)
        fused_img_features = self.projector(fused_img_features)

        deep_img_features = self.projector_base(deep_features)

        if self.adaptive_scale:
            output = deep_img_features + fused_img_features * self.scale_value()
        else:
            output = deep_img_features + fused_img_features
        return output


class ProjectorHighResShallowMinAttnV1(nn.Module):
    def __init__(self, vision_dim, out_dim, dim_head=64, **kwargs):
        super().__init__()
        self.initial_projection_dim = vision_dim * 4
        heads = vision_dim // dim_head

        self.min_attention = MinAttention(q_dim=vision_dim, kv_dim=vision_dim, dim_head=dim_head, heads=heads)
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim)
        )
        self.projector_base = nn.Linear(vision_dim, out_dim, bias=True)

        self.min_attention2 = MinAttention(q_dim=vision_dim, kv_dim=vision_dim, dim_head=dim_head, heads=heads)
        self.projector2 = nn.Sequential(
            nn.Linear(vision_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim)
        )

    def forward(self, vision_input_dict, time_emb=None, **kwargs):
        """
        vision_input_dict: here, this is not a dict, just for the unity of naming
        """
        img_patch_features = vision_input_dict
        shallow_features1, shallow_features2, shallow_features3, deep_features, deep_features_local = img_patch_features
        shallow_features = torch.cat([shallow_features1, shallow_features2, shallow_features3], dim=1) # token concat

        # original code
        fused_img_features = self.min_attention(deep_features_local, deep_features)
        fused_img_features = self.projector(fused_img_features)

        deep_img_features = self.projector_base(deep_features)

        output = deep_img_features + fused_img_features

        # new code part
        fused_img_features2 = self.min_attention2(shallow_features, deep_features)
        fused_img_features2 = self.projector2(fused_img_features2)

        output = torch.cat([deep_img_features, fused_img_features2], dim=1)
        return output