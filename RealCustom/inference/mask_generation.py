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
import torch.nn.functional as F
from einops import rearrange

def mask_generation(
    crossmap_2d_list, selfmap_2d_list=None, 
    target_token=None, mask_scope=None,
    mask_target_h=64, mask_target_w=64,
    mask_mode=["binary"],
):  
    if len(selfmap_2d_list) > 0:
        target_hw_selfmap = mask_target_h * mask_target_w
        selfmap_2ds = []
        for i in range(len(selfmap_2d_list)):
            selfmap_ = selfmap_2d_list[i]
            selfmap_ = F.interpolate(selfmap_, size=(target_hw_selfmap, target_hw_selfmap), mode='bilinear')
            selfmap_2ds.append(selfmap_ )
        selfmap_2ds = torch.cat(selfmap_2ds, dim=1)
        if "selfmap_min_max_per_channel" in mask_mode:
            selfmap_1ds = rearrange(selfmap_2ds, "b c h w -> b c (h w)")
            channel_max_self = torch.max(selfmap_1ds, dim=-1, keepdim=True)[0].unsqueeze(-1)
            channel_min_self = torch.min(selfmap_1ds, dim=-1, keepdim=True)[0].unsqueeze(-1)
            selfmap_2ds = (selfmap_2ds - channel_min_self) / (channel_max_self - channel_min_self + 1e-6)
        elif "selfmap_max_norm" in mask_mode:
            selfmap_1ds = rearrange(selfmap_2ds, "b c h w -> b c (h w)")
            b = selfmap_1ds.size(0)
            batch_max = torch.max(selfmap_1ds.view(b, -1), dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            selfmap_2ds  = selfmap_2ds / (batch_max + 1e-10)

        selfmap_2d = selfmap_2ds.mean(dim=1, keepdim=True)
    else:
        selfmap_2d = None
    
    crossmap_2ds = []
    for i in range(len(crossmap_2d_list)):
        crossmap = crossmap_2d_list[i]
        crossmap = crossmap.mean(dim=1)  # average on head dim
        crossmap = crossmap * target_token.unsqueeze(-1).unsqueeze(-1) # target token valid
        crossmap = crossmap.sum(dim=1, keepdim=True)

        crossmap = F.interpolate(crossmap, size=(mask_target_h, mask_target_w), mode='bilinear')
        crossmap_2ds.append(crossmap)
    crossmap_2ds = torch.cat(crossmap_2ds, dim=1)
    crossmap_1ds = rearrange(crossmap_2ds, "b c h w -> b c (h w)")

    if "max_norm" in mask_mode:
        crossmap_1d_avg = torch.mean(crossmap_1ds, dim=1, keepdim=True)  # [b, 1, (h w)]
        if selfmap_2d is not None:
            crossmap_1d_avg = torch.matmul(selfmap_2d, crossmap_1d_avg.unsqueeze(-1)).squeeze(-1)
        b, c, n = crossmap_1ds.shape
        batch_max = torch.max(crossmap_1d_avg.view(b, -1), dim=-1, keepdim=True)[0].unsqueeze(1)
        crossmap_1d_avg = crossmap_1d_avg / (batch_max + 1e-6)
    elif "min_max_norm" in mask_mode:
        crossmap_1d_avg = torch.mean(crossmap_1ds, dim=1, keepdim=True)  # [b, 1, (h w)]
        if selfmap_2d is not None:
            crossmap_1d_avg = torch.matmul(selfmap_2d, crossmap_1d_avg.unsqueeze(-1)).squeeze(-1)
        b, c, n = crossmap_1ds.shape
        batch_max = torch.max(crossmap_1d_avg.view(b, -1), dim=-1, keepdim=True)[0].unsqueeze(1) # NOTE unsqueeze
        batch_min = torch.min(crossmap_1d_avg.view(b, -1), dim=-1, keepdim=True)[0].unsqueeze(1) # NOTE unsqueeze
        crossmap_1d_avg = (crossmap_1d_avg - batch_min) / (batch_max - batch_min + 1e-6)
    elif "min_max_per_channel" in mask_mode:
        channel_max = torch.max(crossmap_1ds, dim=-1, keepdim=True)[0]
        channel_min = torch.min(crossmap_1ds, dim=-1, keepdim=True)[0]
        crossmap_1ds = (crossmap_1ds - channel_min) / (channel_max - channel_min + 1e-6)
        crossmap_1d_avg = torch.mean(crossmap_1ds, dim=1, keepdim=True)  # [b, 1, (h w)]
        if selfmap_2d is not None:
            crossmap_1d_avg = torch.matmul(selfmap_2d, crossmap_1d_avg.unsqueeze(-1)).squeeze(-1)

        # renormalize to 0-1
        b, c, n = crossmap_1d_avg.shape
        batch_max = torch.max(crossmap_1d_avg.view(b, -1), dim=-1, keepdim=True)[0].unsqueeze(1)
        batch_min = torch.min(crossmap_1d_avg.view(b, -1), dim=-1, keepdim=True)[0].unsqueeze(1)
        crossmap_1d_avg = (crossmap_1d_avg - batch_min) / (batch_max - batch_min + 1e-6)
    else:
        crossmap_1d_avg = torch.mean(crossmap_1ds, dim=1, keepdim=True)  # [b, 1, (h w)]
        
        
    if "threshold" in mask_mode:
        threshold = 1 - mask_scope
        crossmap_1d_avg[crossmap_1d_avg < threshold] = 0.0
        if "binary" in mask_mode:
            crossmap_1d_avg[crossmap_1d_avg > threshold] = 1.0
    else:
        # topk
        topk_num = int(crossmap_1d_avg.size(-1) * mask_scope)
        sort_score, sort_order = crossmap_1d_avg.sort(descending=True, dim=-1)
        sort_topk = sort_order[:, :, :topk_num]
        sort_topk_remain = sort_order[:, :, topk_num:]
        crossmap_1d_avg = crossmap_1d_avg.scatter(2, sort_topk_remain, 0.)
        if "binary" in mask_mode:
            crossmap_1d_avg = crossmap_1d_avg.scatter(2, sort_topk, 1.0)

    crossmap_2d_avg = rearrange(crossmap_1d_avg, "b c (h w) -> b c h w", h=mask_target_h, w=mask_target_w)
    crossmap_2d_avg = crossmap_2d_avg

    output = crossmap_2d_avg.unsqueeze(1)  # torch.Size([4, 1, 60, 64, 64]), The second dimension is the dimension of the number of reference images.
    if output.size(2) == 1: # The dimension of the layer.  
        output = output.squeeze(2)  # If there is only a single dimension, then all layers will share the same mask.

    return output