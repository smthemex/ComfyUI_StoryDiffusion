import os
import math
from typing import List

import torch
import torch.nn as nn
from diffusers.pipelines.controlnet import MultiControlNetModel

from .attention_processor import MaskedIPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, \
    CNAttnProcessor2_0 as CNAttnProcessor


class MSAdapter(torch.nn.Module):
    def __init__(self, pipe, image_proj_model, adapter_modules=None, ckpt_path=None,
                 num_tokens=4, text_tokens=77, max_rn=4, num_dummy_tokens=4,
                 device="cuda", controlnet=None):
        super().__init__()
        self.unet = pipe.unet
        self.pipe = pipe
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.num_tokens = num_tokens
        self.num_dummy_tokens = num_dummy_tokens
        self.text_tokens = text_tokens
        self.max_rn = max_rn
        self.device = device
        self.controlnet = controlnet
        self.cross_attention_dim = self.unet.config.cross_attention_dim
        
        # set attention processor when inference
        if self.adapter_modules is None:
            self.set_ms_adapter()
        
        # dummy image tokens
        self.dummy_image_tokens = nn.Parameter(torch.randn(1, self.num_dummy_tokens, self.cross_attention_dim))
        
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
    
    def load_from_checkpoint(self, ckpt_path: str):
        
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Load state dict for image_proj_model and adapter_modules when using resampler
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        self.adapter_modules.load_state_dict(state_dict["ms_adapter"], strict=False)
        self.load_state_dict({"dummy_image_tokens": state_dict["dummy_image_tokens"]}, strict=False)
        
        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        
        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
        
        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    
    def set_ms_adapter(self, weight_dtype=torch.float16, cache_attention_maps=True):
        # set attention processor
        attn_procs_ = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs_[name] = AttnProcessor()
            else:
                attn_procs_[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=self.num_tokens,
                    text_tokens=self.text_tokens,
                ).to(self.device, dtype=weight_dtype)
        self.unet.set_attn_processor(attn_procs_)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        if self.controlnet is not None:
            if isinstance(self.controlnet, MultiControlNetModel):
                for controlnet in self.controlnet.nets:
                    controlnet.set_attn_processor(
                        CNAttnProcessor(text_tokens=self.text_tokens, num_tokens=self.num_tokens))
            else:
                self.controlnet.set_attn_processor(
                    CNAttnProcessor(text_tokens=self.text_tokens, num_tokens=self.num_tokens))
    
    @torch.inference_mode()
    def get_image_embeds(self, processed_images, image_encoder=None, image_proj_type="linear",
                         image_encoder_type="clip", weight_dtype=torch.float16, use_repo=None):
        # get image embeds
        # processed_images: [bsz, rn, ...]
        if use_repo:
            processed_images = processed_images.view(-1, processed_images.shape[-3], processed_images.shape[-2],
                                                     processed_images.shape[-1])  # (bsz*rn, ...)
        # print(processed_images.shape)  # torch.Size([1, 3, 224, 224])
        
        if image_proj_type == "resampler":
            if use_repo:
                image_embeds = image_encoder(processed_images.to(self.device, dtype=weight_dtype),
                                             output_hidden_states=True).hidden_states[
                    -2]  # (bsz*rn, num_tokens, embedding_dim)
            else:
                
                image_encoder = image_encoder.encode_image
                processed_images.to("cpu")
                image_embeds = image_encoder(processed_images)["penultimate_hidden_states"]
            # print(image_embeds.shape)
        else:
            
            image_embeds = image_encoder(
                processed_images.to(self.device, dtype=weight_dtype)).image_embeds  # (bsz*rn, embedding_dim)
        
        return image_embeds  # [bsz*rn, ...]
    
    def set_scale(self, scale, subject_scales):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
                attn_processor.subject_scales = subject_scales
    
    def enable_psuedo_attention_mask(self, mask_threshold=0.5, start_step=5):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.mask_threshold = mask_threshold
                attn_processor.start_step = start_step
                attn_processor.use_psuedo_attention_mask = True
                attn_processor.need_text_attention_map = True
                attn_processor.attention_maps = []  # clear attention maps
    
    def generate(self,pipe,image_embeds,  prompt_embeds_, negative_prompt_embeds_,grounding_kwargs,cross_attention_kwargs, bsz,height,width,steps,seed,cfg,pooled_prompt_embeds,negative_pooled_prompt_embeds,scale=1.0,
                 num_samples=4,  weight_dtype=torch.float16, subject_scales=None, mask_threshold=None, start_step=5,**kwargs):
       
        self.set_scale(scale, subject_scales)
        if mask_threshold is not None:
            self.enable_psuedo_attention_mask(mask_threshold, start_step)
        
        with (torch.inference_mode()):
            #print(image_embeds.is_cuda,grounding_kwargs),
            image_prompt_embeds = self.image_proj_model(image_embeds, grounding_kwargs=grounding_kwargs) #([2, 16, 2048])
            
            image_prompt_embeds = image_prompt_embeds.view(bsz, -1, image_prompt_embeds.shape[-2],
                                                           image_prompt_embeds.shape[
                                                               -1])  # (bsz, rn, num_tokens, cross_attention_dim)
            image_prompt_embeds = image_prompt_embeds.view(bsz,
                                                           image_prompt_embeds.shape[-3] * image_prompt_embeds.shape[
                                                               -2],
                                                           image_prompt_embeds.shape[
                                                               -1])  # (bsz, total_num_tokens*rn, cross_attention_dim)

            dummy_image_tokens=self.dummy_image_tokens.to(self.device, weight_dtype)
            image_prompt_embeds = torch.cat([dummy_image_tokens, image_prompt_embeds], dim=1)
    
            uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1) #([1, 36, 2048])
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            prompt_embeds_=prompt_embeds_.to(self.device, weight_dtype)#([1, 77, 2048])
            negative_prompt_embeds_=negative_prompt_embeds_.to(self.device, weight_dtype)
            
            generator=torch.Generator(device=self.device).manual_seed(seed)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            
            samples = pipe(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                           generator=generator,
                           prompt_embeds=prompt_embeds,
                           negative_prompt_embeds=negative_prompt_embeds,
                           pooled_prompt_embeds=pooled_prompt_embeds,
                           negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                           cross_attention_kwargs=cross_attention_kwargs,
                           **kwargs
                           )[0]  # torch.Size([1, 4, 64, 64])
            
        return samples

