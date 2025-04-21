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

"""
dinosiglip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and SigLIP.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple
import os
import timm
import torch
from PIL import Image
from einops import rearrange
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from .base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple, return_tuple

import torch.nn as nn
import torchvision

@dataclass
class DinoSigLIPImageTransform:
    dino_image_transform: ImageTransform
    siglip_image_transform: ImageTransform
    is_cobra: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"dino": self.dino_image_transform(img, **kwargs).unsqueeze(0), "siglip": self.siglip_image_transform(img, **kwargs).unsqueeze(0)}



class DinoViTBackbone(VisionBackbone):
    def __init__(self, backbone_name_or_path, image_resize_strategy: str, default_image_size: int = 224, last_n = 2, feature_index = 22,dino_path="") -> None:
        super().__init__(backbone_name_or_path, image_resize_strategy, default_image_size=default_image_size)
        # load from local paths
        self.local_path_ = dino_path#'ckpts/vit_large_patch14_reg4_dinov2.lvd142m/pytorch_model.bin'
        dino_pretrained_cfg = timm.models.create_model(backbone_name_or_path).default_cfg
        if self.local_path_:
            dino_pretrained_cfg['file'] =  self.local_path_

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        self.dino_featurizer: VisionTransformer = timm.create_model(
            backbone_name_or_path, pretrained=True, num_classes=0, img_size=self.default_image_size,
            pretrained_cfg=dino_pretrained_cfg, 
        )
        self.dino_featurizer.eval()

        # Monkey-Patch the `forward()` function of the featurizers to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        # return the output tokens from the `n` last blocks
        print("dino has {} layer intermediate features. ".format(len(self.dino_featurizer.blocks))) # 24 
        # self.dino_featurizer.forward = unpack_tuple(
        #     partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - last_n})
        # )
        if isinstance(feature_index, tuple) or isinstance(feature_index, list):
            feature_index = set(feature_index)
        else:
            feature_index = {feature_index}
        self.dino_featurizer.forward = return_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n=feature_index)
        )

        # Get Configs for _both_ Featurizers =>> Note :: Override default image size for larger resolution models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize *both* Transforms
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)

        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
            assert isinstance(dino_resize_transform := default_dino_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=dino_resize_transform.interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )

            self.dino_transform = dino_transform
        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values, device="cpu", input_dtype_new=None) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        # b, c , h , w : 0-1
        t_tensors = []
        for pixel_value in pixel_values:
            t_tensors.append(self.dino_transform(pixel_value).unsqueeze(0))
        t_tensors = torch.cat(t_tensors, dim=0).to(device)
        if input_dtype_new is not None:
            t_tensors = t_tensors.to(input_dtype_new)

        t_tensors_list = self.dino_featurizer(t_tensors)
        return t_tensors_list
        
    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16


class DinoEncoder(nn.Module):
    def __init__(self, backbone_name_or_path, image_resize_strategy: str, default_image_size: int = 224, feature_index = 22) -> None:
        super().__init__()

        self.image_encoder = DinoViTBackbone(backbone_name_or_path, image_resize_strategy, default_image_size, feature_index)
        self.to_pil = torchvision.transforms.ToPILImage()

    def forward(self, image_tensor, device="cpu", input_dtype_new=torch.float32): # input image size = 768
        pixel_values = []

        for image_tensor_i in image_tensor:
            pixel_values.append(self.to_pil(image_tensor_i))
        
        embeddings_dino_list = self.image_encoder(pixel_values, device, input_dtype_new)
        if len(embeddings_dino_list) == 1:
            embeddings_dino_list = embeddings_dino_list[0]
        return embeddings_dino_list

class DinoEncoderV2(nn.Module):
    def __init__(self, backbone_name_or_path, image_resize_strategy: str, default_image_size: int = 224, feature_index = 22) -> None:
        super().__init__()

        self.image_encoder = DinoViTBackbone(backbone_name_or_path, image_resize_strategy, default_image_size, feature_index)
        self.to_pil = torchvision.transforms.ToPILImage()
    
    def get_fsdp_wrapping_policy(self):
        return self.image_encoder.get_fsdp_wrapping_policy()

    def forward(self, image_tensor_dict, device="cpu", input_dtype_new=torch.float32): 
        image_tensor = image_tensor_dict["images_ref"]
        
        output_dict = {}
        pixel_values = []

        for image_tensor_i in image_tensor:
            pixel_values.append(self.to_pil(image_tensor_i))
        
        embeddings_dino_list = self.image_encoder(pixel_values, device, input_dtype_new)
        if len(embeddings_dino_list) == 1:
            embeddings_dino_list = embeddings_dino_list[0]
        output_dict["img_patch_features"] = embeddings_dino_list
        return output_dict

class DinoEncoderV2_Canny(nn.Module):
    def __init__(self, backbone_name_or_path, image_resize_strategy: str, default_image_size: int = 224, feature_index = 22) -> None:
        super().__init__()

        self.image_encoder = DinoViTBackbone(backbone_name_or_path, image_resize_strategy, default_image_size, feature_index)
        self.to_pil = torchvision.transforms.ToPILImage()
    
    def get_fsdp_wrapping_policy(self):
        return self.image_encoder.get_fsdp_wrapping_policy()

    def forward(self, image_tensor_dict, device="cpu", input_dtype_new=torch.float32): 
        image_canny = image_tensor_dict["images_canny"]
        
        output_dict = {}
        pixel_values = []

        for image_tensor_i in image_canny:
            pixel_values.append(self.to_pil(image_tensor_i))
        
        embeddings_dino_list = self.image_encoder(pixel_values, device, input_dtype_new)
        if len(embeddings_dino_list) == 1:
            embeddings_dino_list = embeddings_dino_list[0]
        output_dict["img_patch_features"] = embeddings_dino_list
        return output_dict