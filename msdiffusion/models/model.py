import os
import math
from typing import List

import torch
import torch.nn as nn
from diffusers.pipelines.controlnet import MultiControlNetModel

from .attention_processor import MaskedIPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, \
    CNAttnProcessor2_0 as CNAttnProcessor


class MSAdapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules=None, ckpt_path=None,
                 num_tokens=4, text_tokens=77, max_rn=4, num_dummy_tokens=4,
                 device="cuda", controlnet=None):
        super().__init__()
        self.unet = unet
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
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "ms_adapter.bin")
        
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
        attn_procs = {}
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
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=self.num_tokens,
                    text_tokens=self.text_tokens,
                ).to(self.device, dtype=weight_dtype)
        self.unet.set_attn_processor(attn_procs)
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
                print(processed_images.shape,1233333)
                image_encoder = image_encoder.encode_image
                processed_images.to("cpu")
                image_embeds = image_encoder(processed_images)["penultimate_hidden_states"]
            print(image_embeds.shape,1234567)
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
    
    def generate(self, pipe, pil_images=None, processed_images=None, prompt=None, negative_prompt=None, scale=1.0,
                 num_samples=4, seed=None, guidance_scale=7.5, num_inference_steps=50, image_processor=None,
                 image_encoder=None, image_proj_type="linear", image_encoder_type="clip", weight_dtype=torch.float16,
                 boxes=None, phrases=None, drop_grounding_tokens=None, phrase_idxes=None,
                 eot_idxes=None, height=1024, width=1024, subject_scales=None, mask_threshold=None, start_step=5,
                 use_repo=None,
                 **kwargs):
        # generate images (validation&inference)
        self.pipe = pipe
        self.set_scale(scale, subject_scales)
        if mask_threshold is not None:
            self.enable_psuedo_attention_mask(mask_threshold, start_step)
        
        # pil_images: [[xxx, xxx, xxx], [xxx, xxx, xxx], ...]
        bsz = len(pil_images)  # only support bsz=1 now
        if use_repo:
            if processed_images is None:
                # write in this way to promise it can be extended to batch in the future
                processed_images = []
                for pil_image in pil_images:
                    processed_image = image_processor(images=pil_image, return_tensors="pt").pixel_values
                    processed_images.append(processed_image)
                processed_images = torch.stack(processed_images, dim=0)
       
        num_prompts = bsz
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"  # duplicate
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        cross_attention_kwargs = None
        grounding_kwargs = None
        if boxes is not None:
            boxes = torch.tensor(boxes).to(self.device, weight_dtype)
            print(boxes.shape,123)
            if phrases is not None:
                drop_grounding_tokens = drop_grounding_tokens if drop_grounding_tokens is not None else [0] * bsz
                batch_boxes = boxes.view(bsz * boxes.shape[1], -1).to(self.device)
                # write in this way to promise it can be extended to batch in the future
                phrase_input_ids = []
                for phrase in phrases:
                    phrase_input_id = pipe.tokenizer(phrase, max_length=pipe.tokenizer.model_max_length,
                                                     padding="max_length", truncation=True,
                                                     return_tensors="pt").input_ids
                    phrase_input_ids.append(phrase_input_id)
                phrase_input_ids = torch.stack(phrase_input_ids)
                phrase_input_ids = phrase_input_ids.view(-1, phrase_input_ids.shape[-1])
                phrase_embeds = pipe.text_encoder(phrase_input_ids.to(self.device)).pooler_output
                print(phrase_embeds.shape,batch_boxes.shape) # torch.Size([2, 768]) torch.Size([2, 4])
                grounding_kwargs = {"boxes": batch_boxes, "phrase_embeds": phrase_embeds,
                                    "drop_grounding_tokens": drop_grounding_tokens}
            else:
                grounding_kwargs = None
            boxes = torch.repeat_interleave(boxes, repeats=num_samples, dim=0)
            uncond_boxes = torch.zeros_like(boxes)
            boxes = torch.cat([uncond_boxes, boxes], dim=0)
            cross_attention_kwargs = {"boxes": boxes}
        
        if phrase_idxes is not None:
            phrase_idxes = torch.tensor(phrase_idxes).to(self.device, torch.int)
            eot_idxes = torch.tensor(eot_idxes).to(self.device, torch.int)
            phrase_idxes = torch.repeat_interleave(phrase_idxes, repeats=num_samples, dim=0)
            eot_idxes = torch.repeat_interleave(eot_idxes, repeats=num_samples, dim=0)
            uncond_phrase_idxes = torch.zeros_like(phrase_idxes)
            uncond_eot_idxes = torch.zeros_like(eot_idxes)
            phrase_idxes = torch.cat([uncond_phrase_idxes, phrase_idxes], dim=0)
            eot_idxes = torch.cat([uncond_eot_idxes, eot_idxes], dim=0)
            if cross_attention_kwargs is None:
                cross_attention_kwargs = {"phrase_idxes": phrase_idxes, "eot_idxes": eot_idxes}
            else:
                cross_attention_kwargs["phrase_idxes"] = phrase_idxes
                cross_attention_kwargs["eot_idxes"] = eot_idxes
        
        with (torch.inference_mode()):
            #print(processed_images.shape)
            image_embeds = self.get_image_embeds(processed_images, image_encoder, image_proj_type=image_proj_type,
                                                 image_encoder_type=image_encoder_type, weight_dtype=weight_dtype,
                                                 use_repo=use_repo)
            del image_encoder
            torch.cuda.empty_cache()
            # print(image_embeds.shape) #torch.Size([2, 257, 1664])
            if not use_repo:
               # print(image_embeds.device,self.device,)
                image_embeds=image_embeds.clone().detach().to(self.device, weight_dtype)
                #print(image_embeds.device)
            image_prompt_embeds = self.image_proj_model(image_embeds, grounding_kwargs=grounding_kwargs)
           
            del image_embeds
            torch.cuda.empty_cache()
            image_prompt_embeds = image_prompt_embeds.view(bsz, -1, image_prompt_embeds.shape[-2],
                                                           image_prompt_embeds.shape[
                                                               -1])  # (bsz, rn, num_tokens, cross_attention_dim)
            image_prompt_embeds = image_prompt_embeds.view(bsz,
                                                           image_prompt_embeds.shape[-3] * image_prompt_embeds.shape[
                                                               -2],
                                                           image_prompt_embeds.shape[
                                                               -1])  # (bsz, total_num_tokens*rn, cross_attention_dim)
            image_prompt_embeds = torch.cat([self.dummy_image_tokens, image_prompt_embeds], dim=1)
            
            uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            
            prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
           
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
            height=height,
            width=width,
            **kwargs,
        ).images
        
        return images
