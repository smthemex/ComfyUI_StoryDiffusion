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
import torchvision
import torch.nn as nn
from einops import rearrange
import os
from .sigclip import SigLIPViTBackbone
from .dino import DinoViTBackbone
cur_path = os.path.dirname(os.path.abspath(__file__))
import folder_paths



class ShallowDeepSiglipDinoEncoder(nn.Module):
    def __init__(self, siglip_config={}, dino_config={}):
        super().__init__()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.image_encoder_siglip = SigLIPViTBackbone(**siglip_config)
        self.image_encoder_dino = DinoViTBackbone(**dino_config)
    
    def forward(self, image_tensor, device="cpu"):
        bs = image_tensor.size(0)        
        # tensor 转 PIL
        pixel_values = []
        for image_tensor_i in image_tensor:
            pixel_values.append(self.to_pil(image_tensor_i))

        embeddings = []
        embeddings_siglip_list = self.image_encoder_siglip(pixel_values, device)
        embeddings_dino_list = self.image_encoder_dino(pixel_values, device)
        for embeddings_siglip_i, embeddings_dino_i in zip(embeddings_siglip_list, embeddings_dino_list):
            embeddings_i = torch.cat([embeddings_siglip_i, embeddings_dino_i], dim=-1) # channel concat
            embeddings.append(embeddings_i)

        return embeddings

# The default is to use double the image size, i.e., 768x768.
class ShallowDeepPatchfySiglipDinoEncoder(nn.Module):
    def __init__(self, siglip_config={}, dino_config={}, patchfy_scale=2, default_image_size=384,siglip_path="",dino_path=""):
        super().__init__()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.siglip_path=siglip_path
        self.dino_path=dino_path
        self.image_encoder_siglip = SigLIPViTBackbone(**siglip_config,siglip_path=self.siglip_path)
        self.image_encoder_dino = DinoViTBackbone(**dino_config,dino_path=self.dino_path)

        self.patchfy = (patchfy_scale > 1)
        self.patchfy_scale = patchfy_scale
        self.default_image_size = default_image_size
    
    def forward(self, image_tensor, device="cpu", **kwargs): # input image size = 768
        image_tensor = image_tensor["image_ref"] # this is a dict
        bs = image_tensor.size(0)

        if self.patchfy:
            image_local = rearrange(image_tensor, "b c (h hp) (w wp) -> (b hp wp) c h w", hp=self.patchfy_scale, wp=self.patchfy_scale)
            image_global = torch.nn.functional.interpolate(image_tensor, size=(self.default_image_size, self.default_image_size), mode='bilinear', align_corners=True)
        
            # tensor 转 PIL
            pixel_values_local, pixel_values_global = [], []
            for image_tensor_i in image_local:
                pixel_values_local.append(self.to_pil(image_tensor_i.to(torch.float)))
            for image_tensor_i in image_global:
                pixel_values_global.append(self.to_pil(image_tensor_i.to(torch.float)))

            embeddings = []
            embeddings_siglip_list = self.image_encoder_siglip(pixel_values_global, device)
            embeddings_dino_list = self.image_encoder_dino(pixel_values_global, device)
            for embeddings_siglip_i, embeddings_dino_i in zip(embeddings_siglip_list, embeddings_dino_list):
                embeddings_i = torch.cat([embeddings_siglip_i, embeddings_dino_i], dim=-1) # channel concat
                embeddings.append(embeddings_i)
            
            embeddings_local_siglip_deep = self.image_encoder_siglip(pixel_values_local, device)[-1]
            embeddings_local_dino_deep = self.image_encoder_dino(pixel_values_local, device)[-1]
            embeddings_local_deep = torch.cat([embeddings_local_siglip_deep, embeddings_local_dino_deep], dim=-1)

            embeddings_local_deep = rearrange(embeddings_local_deep, "(b hp wp) l c -> b (l hp wp) c", hp=self.patchfy_scale, wp=self.patchfy_scale)

            embeddings.append(embeddings_local_deep)
        
        else:
            # tensor 转 PIL
            pixel_values = []
            for image_tensor_i in image_tensor:
                pixel_values.append(self.to_pil(image_tensor_i))
            
            embeddings = []
            embeddings_siglip_list = self.image_encoder_siglip(pixel_values, device)
            embeddings_dino_list = self.image_encoder_dino(pixel_values, device)
            for embeddings_siglip_i, embeddings_dino_i in zip(embeddings_siglip_list, embeddings_dino_list):
                # 逐层concat的方式
                embeddings_i = torch.cat([embeddings_siglip_i, embeddings_dino_i], dim=-1) # channel concat
                embeddings.append(embeddings_i)
        
        if len(embeddings) == 1:
            embeddings = embeddings[0]
        return embeddings


class ShallowDeepPatchfySiglipDinoEncoder_v2(nn.Module):
    def __init__(self, siglip_config={}, dino_config={}, patchfy_scale=2, default_image_size=384,siglip_path="",dino_path=""):
        super().__init__()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.siglip_path=siglip_path
        self.dino_path=dino_path
        self.image_encoder_siglip = SigLIPViTBackbone(**siglip_config,siglip_path=self.siglip_path)
        self.image_encoder_dino = DinoViTBackbone(**dino_config,dino_path=self.dino_path)
        
        self.patchfy = (patchfy_scale > 1)
        self.patchfy_scale = patchfy_scale
        self.default_image_size = default_image_size
    
    def forward(self, image_tensor_dict, device="cpu", **kwargs): # input image size = 768
        image_tensor = image_tensor_dict["image_ref"]
        bs = image_tensor.size(0)

        if self.patchfy:
            image_local = rearrange(image_tensor, "b c (h hp) (w wp) -> (b hp wp) c h w", hp=self.patchfy_scale, wp=self.patchfy_scale)
            image_global = torch.nn.functional.interpolate(image_tensor, size=(self.default_image_size, self.default_image_size), mode='bilinear', align_corners=True)
        
            pixel_values_local, pixel_values_global = [], []
            for image_tensor_i in image_local:
                pixel_values_local.append(self.to_pil(image_tensor_i.to(torch.float32)))
            for image_tensor_i in image_global:
                pixel_values_global.append(self.to_pil(image_tensor_i.to(torch.float32)))

            embeddings = []
            embeddings_siglip_list = self.image_encoder_siglip(pixel_values_global, device)
            embeddings_dino_list = self.image_encoder_dino(pixel_values_global, device)
            for embeddings_siglip_i, embeddings_dino_i in zip(embeddings_siglip_list, embeddings_dino_list):
                embeddings_i = torch.cat([embeddings_siglip_i, embeddings_dino_i], dim=-1) # channel concat
                embeddings.append(embeddings_i)
            
            embeddings_local_siglip_deep = self.image_encoder_siglip(pixel_values_local, device)[-1]
            embeddings_local_dino_deep = self.image_encoder_dino(pixel_values_local, device)[-1]
            embeddings_local_deep = torch.cat([embeddings_local_siglip_deep, embeddings_local_dino_deep], dim=-1)

            embeddings_local_deep = rearrange(embeddings_local_deep, "(b hp wp) l c -> b (l hp wp) c", hp=self.patchfy_scale, wp=self.patchfy_scale)

            embeddings.append(embeddings_local_deep)
        
        else:
            # tensor 转 PIL
            pixel_values = []
            for image_tensor_i in image_tensor:
                pixel_values.append(self.to_pil(image_tensor_i))
            
            embeddings = []
            embeddings_siglip_list = self.image_encoder_siglip(pixel_values, device)
            embeddings_dino_list = self.image_encoder_dino(pixel_values, device)
            for embeddings_siglip_i, embeddings_dino_i in zip(embeddings_siglip_list, embeddings_dino_list):
                embeddings_i = torch.cat([embeddings_siglip_i, embeddings_dino_i], dim=-1) # channel concat
                embeddings.append(embeddings_i)
        
        if len(embeddings) == 1:
            embeddings = embeddings[0]
        return embeddings
