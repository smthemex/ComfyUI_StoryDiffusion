 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import comfy.utils
import node_helpers
import torch
import os
from omegaconf import OmegaConf
try:
    from .diffusers_wrapper import QwenImageTransformer2DModel    
except:
    from diffusers import QwenImageTransformer2DModel
    
from .diffusers_wrapper import QwenImageEditPipeline,QwenImagePipeline
from transformers.modeling_utils import no_init_weights
from diffusers import  FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel
import torch
import torch.nn as nn
from safetensors.torch import safe_open

def load_quwen_image(cpu_offload,cpu_offload_blocks,no_pin_memory, dir_path,repo,unet_path=None,gguf_path=None,quantize_mode="fp8",lora_path=None):
    edit_mode=False
    if repo is not None:
        model_id=os.path.join(dir_path,"qwen_image/Qwen-Image-Edit") if "edit" in repo.lower() else os.path.join(dir_path,"qwen_image/Qwen-Image")
        vae=OmegaConf.load(os.path.join(dir_path, "qwen_image/Qwen-Image-Edit/vae/config.json")) if "edit" in repo.lower() else OmegaConf.load(os.path.join(dir_path, "qwen_image/Qwen-Image/vae/config.json"))
        edit_mode=True if "edit" in repo.lower() else False
    elif gguf_path:
        model_id=os.path.join(dir_path,"qwen_image/Qwen-Image-Edit") if "edit" in gguf_path.lower() else os.path.join(dir_path,"qwen_image/Qwen-Image")
        vae=OmegaConf.load(os.path.join(dir_path, "qwen_image/Qwen-Image-Edit/vae/config.json")) if "edit" in gguf_path.lower() else OmegaConf.load(os.path.join(dir_path, "qwen_image/Qwen-Image/vae/config.json"))
        edit_mode=True if "edit" in gguf_path.lower() else False
    elif unet_path:
        model_id=os.path.join(dir_path,"qwen_image/Qwen-Image-Edit") if "edit" in unet_path.lower() else os.path.join(dir_path,"qwen_image/Qwen-Image")
        vae=OmegaConf.load(os.path.join(dir_path, "qwen_image/Qwen-Image-Edit/vae/config.json")) if "edit" in unet_path.lower() else OmegaConf.load(os.path.join(dir_path, "qwen_image/Qwen-Image/vae/config.json"))
        edit_mode=True if "edit" in unet_path.lower() else False



    if repo is not None :
        if "float" in repo.lower():
            from dfloat11 import DFloat11Model
            
            with no_init_weights():
                unet_config=QwenImageTransformer2DModel.load_config(
                        model_id, subfolder="transformer",)
                transformer = QwenImageTransformer2DModel.from_config(unet_config).to(torch.bfloat16)

            DFloat11Model.from_pretrained(
                repo,
                device="cpu",
                cpu_offload=cpu_offload,
                cpu_offload_blocks=cpu_offload_blocks,
                pin_memory=not no_pin_memory,
                bfloat16_model=transformer,
            )
        

    elif  gguf_path is not None:
        from diffusers import  GGUFQuantizationConfig
        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            config=os.path.join(model_id, "transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
    else:
        if unet_path is None:
            if quantize_mode=="fp8":
                transformer = QwenImageTransformer2DModel.from_single_file(unet_path, config=os.path.join(model_id,"transformer/config.json"),
                                                                                torch_dtype=torch.bfloat16)
            else:
                from safetensors.torch import load_file
                t_state_dict=load_file(unet_path)
                unet_config = QwenImageTransformer2DModel.load_config(os.path.join(model_id,"transformer/config.json"))
                transformer = QwenImageTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
                transformer.load_state_dict(t_state_dict, strict=False)
                del t_state_dict
   
    if lora_path is not None:
        print(f"load lora weights : {lora_path}")
        scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        if not edit_mode :
            
            if  "lightning" in lora_path.lower():
                # transformer = load_and_merge_lora_weight_from_safetensors(transformer, lora_path)
                pipeline = QwenImagePipeline.from_pretrained(
                    model_id, transformer=transformer,scheduler=scheduler,VAE=vae, torch_dtype=torch.bfloat16,
                )
            else:
                pipeline = QwenImagePipeline.from_pretrained(
                    model_id, transformer=transformer,VAE=vae, torch_dtype=torch.bfloat16,
                )
                
        else:
            if "lightning" in lora_path.lower():
                pipeline = QwenImageEditPipeline.from_pretrained(
                model_id,transformer=transformer, VAE=vae,scheduler=scheduler, torch_dtype=torch.bfloat16,
                )
            else:
                 pipeline = QwenImageEditPipeline.from_pretrained(
                model_id,transformer=transformer, VAE=vae, torch_dtype=torch.bfloat16,
                )
        pipeline.load_lora_weights(lora_path,weight_name= os.path.basename(lora_path))
    else:
        if not edit_mode :
            pipeline = QwenImagePipeline.from_pretrained(
                model_id, transformer=transformer, VAE=vae, torch_dtype=torch.bfloat16,
            )
        else:
            pipeline = QwenImageEditPipeline.from_pretrained(
                model_id,transformer=transformer, VAE=vae, torch_dtype=torch.bfloat16,
            )
    
    from diffusers.hooks import apply_group_offloading
    onload_device = torch.device("cuda")
    apply_group_offloading(pipeline.transformer, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=1)
    #pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=None)
    # use_mmgp="HighRAM_LowVRAM"
    # if use_mmgp!="None":
    #     from mmgp import offload, profile_type
    #     pipeline.to("cpu")
    #     if use_mmgp=="VerylowRAM_LowVRAM":
    #         offload.profile(pipeline, profile_type.VerylowRAM_LowVRAM,quantizeTransformer=False)
    #     elif use_mmgp=="LowRAM_LowVRAM":  
    #         offload.profile(pipeline, profile_type.LowRAM_LowVRAM,quantizeTransformer=False)
    #     elif use_mmgp=="LowRAM_HighVRAM":
    #         offload.profile(pipeline, profile_type.LowRAM_HighVRAM,quantizeTransformer=False)
    #     elif use_mmgp=="HighRAM_LowVRAM":
    #         offload.profile(pipeline, profile_type.HighRAM_LowVRAM,quantizeTransformer=False)
    #     elif use_mmgp=="HighRAM_HighVRAM":
    #         offload.profile(pipeline, profile_type.HighRAM_HighVRAM,quantizeTransformer=False)
    return pipeline,edit_mode


def get_emb_data(clip,vae,prompt_list,image_list,role_list,)  : #image_list[ tensor,]
    ref_latent = None
    if image_list is None:
        images = []
    else:
        if len(role_list)==1:
            image=image_list[0]
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])
        else:
            ref_latents = []
            ref_iamges = []
            for image in image_list:
                samples = image.movedim(-1, 1)
                total = int(1024 * 1024)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale()
                image = s.movedim(1, -1)
                images = [image[:, :, :, :3]]
                ref_iamges.append(images)
                if vae is not None:
                    ref_latent = vae.encode(image[:, :, :, :3])
                    ref_latents.append(ref_latent)
                
    role_emb_dict={}              
    if len(role_list)==1: #prompt_list[[prompt]*len(prompt_list)]
        pos_cond_list=[]
        for prompt in prompt_list[0]:
            tokens = clip.tokenize(prompt, images=images)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            if ref_latent is not None: #传递latent
                conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True) #[[tensor, dict]]
            
            prompt_embeds=conditioning[0][0]    
            #print(prompt_embeds.shape) #torch.Size([1, 1377, 3584])
            
            pol_dict=conditioning[0][1]["reference_latents"][0] if ref_latent is not None else None   
            # print(pol_dict.shape) torch.Size([1, 16, 1, 128, 128])
            pos_cond_list.append((prompt_embeds,pol_dict))
        role_emb_dict[role_list[0]]=pos_cond_list # {role:[(prompt_embeds,reference_latents),...]}
        return role_emb_dict
    else:
        for role,role_text_list,img,lat in zip(role_list,prompt_list,ref_iamges,ref_latents): #[[prompt],[prompt]]
            pos_cond_list=[]
            for prompt in role_text_list:
                tokens = clip.tokenize(prompt, images=img)
                conditioning = clip.encode_from_tokens_scheduled(tokens)
                if ref_latent is not None:
                    conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [lat]}, append=True)
                
                prompt_embeds=conditioning[0][0]
                pol_dict=conditioning[0][1]["reference_latents"][0] if ref_latent is not None else None     
             
                pos_cond_list.append((prompt_embeds,pol_dict))
            role_emb_dict[role]=pos_cond_list
    
        return role_emb_dict


def infer_qwen_image_edit(pipeline, image, prompt_embeds, prompt_embeds_mask,negative_prompt_embeds,negative_prompt_embeds_mask,seed, true_cfg_scale,num_inference_steps,image_latents,edit_mode,latents):

    _,_,height, width=latents.size()
    inputs = {
        "image": image,
        "prompt": None,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": None,
        "num_inference_steps": num_inference_steps,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "image_latents":image_latents, 
        "height": height*8 if not edit_mode else None,
        "width": width*8 if not edit_mode else None,      
 
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images
        #print(output_image.shape)
        
        #output_image.save("output_path.png")  # Save the output image for debugging

    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
    return output_image

def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha


def load_and_merge_lora_weight(
    model: nn.Module,
    lora_state_dict: dict,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    is_native_weight = any("diffusion_model." in key for key in lora_state_dict)
    for key, value in model.named_parameters():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        if lora_down_name in lora_state_dict:
            lora_down = lora_state_dict[lora_down_name]
            lora_up = lora_state_dict[lora_up_name]
            lora_alpha = float(lora_state_dict[lora_alpha_name])
            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank
            assert lora_up.dtype == torch.float32
            assert lora_down.dtype == torch.float32
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
            value.data = (value.data + delta_W).type_as(value.data)
    return model


def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    lora_state_dict = {}
    with safe_open(lora_weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    model = load_and_merge_lora_weight(
        model, lora_state_dict, lora_down_key, lora_up_key
    )
    return model
