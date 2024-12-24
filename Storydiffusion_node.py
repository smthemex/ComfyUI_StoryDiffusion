# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import gc
import logging
import numpy as np
import torch
import os
from PIL import ImageFont,Image
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline,EulerDiscreteScheduler, UNet2DConditionModel,UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
import datetime
import folder_paths
from comfy.clip_vision import load as clip_load
from comfy.model_management import total_vram

from .utils.utils import get_comic
from .utils.load_models_utils import load_models
from .model_loader_utils import  (story_maker_loader,kolor_loader,phi2narry,
                                  extract_content_from_brackets,narry_list,remove_punctuation_from_strings,phi_list,center_crop_s,center_crop,
                                  narry_list_pil,setup_seed,find_directories,
                                  apply_style,get_scheduler,apply_style_positive,SD35Wrapper,load_images_list,
                                  nomarl_upscale,SAMPLER_NAMES,SCHEDULER_NAMES,lora_lightning_list,pre_checkpoint,get_easy_function,sd35_loader)
from .utils.gradio_utils import cal_attn_indice_xl_effcient_memory,is_torch2_available,process_original_prompt,get_ref_character,character_to_dict
from .ip_adapter.attention_processor import IPAttnProcessor2_0
if is_torch2_available():
    from .utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from .utils.gradio_utils import AttnProcessor
import torch.nn.functional as F
import copy
global total_count, attn_count, cur_step, mask1024, mask4096, attn_procs, unet
global sa32, sa64
global write
global height_s, width_s

import transformers
transformers_v=float(transformers.__version__.rsplit(".",1)[0])

photomaker_dir=os.path.join(folder_paths.models_dir, "photomaker")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))

fonts_path = os.path.join(dir_path, "fonts")
fonts_lists = os.listdir(fonts_path)

base_pt = os.path.join(photomaker_dir,"pt")
if not os.path.exists(base_pt):
    os.makedirs(base_pt)
    
def set_attention_processor(unet, id_length, is_ipadapter=False):
    global attn_procs
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet.device, dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))

def load_single_character_weights(unet, filepath):
    """
    从指定文件中加载权重到 attention_processor 类的 id_bank 中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重文件的路径。
    """
    # 使用torch.load来读取权重
    weights_to_load = torch.load(filepath, map_location=torch.device("cpu"))
    weights_to_load.eval()
    character = weights_to_load["character"]
    description = weights_to_load["description"]
    #print(character)
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 转移权重到GPU（如果GPU可用的话）并赋值给id_bank
            attn_processor.id_bank[character] = {}
            for step_key in weights_to_load[attn_name].keys():

                attn_processor.id_bank[character][step_key] = [
                    tensor.to(unet.device)
                    for tensor in weights_to_load[attn_name][step_key]
                ]
    print("successsfully,load_single_character_weights")

def load_character_files_on_running(unet, character_files: str):
    if character_files == "":
        return False
    weights_list = os.listdir(character_files)#获取路径下的权重列表
    #character_files_arr = character_files.splitlines()
    for character_file in weights_list:
        path_cur=os.path.join(character_files,character_file)
        load_single_character_weights(unet, path_cur)
    return True
def save_single_character_weights(unet, character, description, filepath):
    """
    保存 attention_processor 类中的 id_bank GPU Tensor 列表到指定文件中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重要保存到的文件路径。
    """
    weights_to_save = {}
    weights_to_save["description"] = description
    weights_to_save["character"] = character
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 将每个 Tensor 转到 CPU 并转为列表，以确保它可以被序列化
            #print(attn_name, attn_processor)
            weights_to_save[attn_name] = {}
            for step_key in attn_processor.id_bank[character].keys():
                weights_to_save[attn_name][step_key] = [
                    tensor.cpu()
                    for tensor in attn_processor.id_bank[character][step_key]
                ]
    # 使用torch.save保存权重
    torch.save(weights_to_save, filepath)
    
def save_results(unet):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    weight_folder_name =os.path.join(base_pt,f"{timestamp}")
    #创建文件夹
    if not os.path.exists(weight_folder_name):
        os.makedirs(weight_folder_name)
    global character_dict
    for char in character_dict:
        description = character_dict[char]
        save_single_character_weights(unet,char,description,os.path.join(weight_folder_name, f'{char}.pt'))
        
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
            id_length=4,
            device=device,
            dtype=torch.float16,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        # un_cond_hidden_states, cond_hidden_states = hidden_states.chunk(2)
        # un_cond_hidden_states = self.__call2__(attn, un_cond_hidden_states,encoder_hidden_states,attention_mask,temb)
        # 生成一个0到1之间的随机数
        global total_count, attn_count, cur_step, indices1024, indices4096
        global sa32, sa64
        global write
        global height_s, width_s
        global character_dict
        global  character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals, cur_character
        if attn_count == 0 and cur_step == 0:
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height_s,
                width_s,
                device=self.device,
                dtype=self.dtype,
            )
        if write:
            assert len(cur_character) == 1
            if hidden_states.shape[1] == (height_s // 32) * (width_s // 32):
                indices = indices1024
            else:
                indices = indices4096
            # print(f"white:{cur_step}")
            total_batch_size, nums_token, channel = hidden_states.shape
            img_nums = total_batch_size // 2
            hidden_states = hidden_states.reshape(-1, img_nums, nums_token, channel)
            # print(img_nums,len(indices),hidden_states.shape,self.total_length)
            if cur_character[0] not in self.id_bank:
                self.id_bank[cur_character[0]] = {}
            self.id_bank[cur_character[0]][cur_step] = [
                hidden_states[:, img_ind, indices[img_ind], :]
                .reshape(2, -1, channel)
                .clone()
                for img_ind in range(img_nums)
            ]
            hidden_states = hidden_states.reshape(-1, nums_token, channel)
            # self.id_bank[cur_step] = [hidden_states[:self.id_length].clone(), hidden_states[self.id_length:].clone()]
        else:
            # encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),self.id_bank[cur_step][1].to(self.device)))
            # TODO: ADD Multipersion Control
            encoder_arr = []
            for character in cur_character:
                encoder_arr = encoder_arr + [
                    tensor.to(self.device)
                    for tensor in self.id_bank[character][cur_step]
                ]
        # 判断随机数是否大于0.5
        if cur_step < 1:
            hidden_states = self.__call2__(
                attn, hidden_states, None, attention_mask, temb
            )
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            # print(f"hidden state shape {hidden_states.shape[1]}")
            if random_number > rand_num:
                if hidden_states.shape[1] == (height_s // 32) * (width_s // 32):
                    indices = indices1024
                else:
                    indices = indices4096
                # print("before attention",hidden_states.shape,attention_mask.shape,encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
                if write:
                    total_batch_size, nums_token, channel = hidden_states.shape
                    img_nums = total_batch_size // 2
                    hidden_states = hidden_states.reshape(
                        -1, img_nums, nums_token, channel
                    )
                    encoder_arr = [
                        hidden_states[:, img_ind, indices[img_ind], :].reshape(
                            2, -1, channel
                        )
                        for img_ind in range(img_nums)
                    ]
                    for img_ind in range(img_nums):
                        # print(img_nums)
                        # assert img_nums != 1
                        img_ind_list = [i for i in range(img_nums)]
                        # print(img_ind_list,img_ind)
                        img_ind_list.remove(img_ind)
                        # print(img_ind,invert_character_index_dict[img_ind])
                        # print(character_index_dict[invert_character_index_dict[img_ind]])
                        # print(img_ind_list)
                        # print(img_ind,img_ind_list)
                        encoder_hidden_states_tmp = torch.cat(
                            [encoder_arr[img_ind] for img_ind in img_ind_list]
                            + [hidden_states[:, img_ind, :, :]],
                            dim=1,
                        )

                        hidden_states[:, img_ind, :, :] = self.__call2__(
                            attn,
                            hidden_states[:, img_ind, :, :],
                            encoder_hidden_states_tmp,
                            None,
                            temb,
                        )
                else:
                    _, nums_token, channel = hidden_states.shape
                    # img_nums = total_batch_size // 2
                    # encoder_hidden_states = encoder_hidden_states.reshape(-1,img_nums,nums_token,channel)
                    hidden_states = hidden_states.reshape(2, -1, nums_token, channel)

                    # encoder_arr = [encoder_hidden_states[:,img_ind,indices[img_ind],:].reshape(2,-1,channel) for img_ind in range(img_nums)]
                    encoder_hidden_states_tmp = torch.cat(
                        encoder_arr + [hidden_states[:, 0, :, :]], dim=1
                    )

                    hidden_states[:, 0, :, :] = self.__call2__(
                        attn,
                        hidden_states[:, 0, :, :],
                        encoder_hidden_states_tmp,
                        None,
                        temb,
                    )
                hidden_states = hidden_states.reshape(-1, nums_token, channel)
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height_s,
                width_s,
                device=self.device,
                dtype=self.dtype,
            )

        return hidden_states

    def __call2__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height_s, width_s = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height_s * width_s
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        # else:
        #     encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height_s, width_s
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def process_generation(
        pipe,
        upload_images,
        model_type,
        _num_steps,
        style_name,
        denoise_or_ip_sacle,
        _style_strength_ratio,
        cfg,
        seed_,
        id_length,
        general_prompt,
        negative_prompt,
        prompt_array,
        width,
        height,
        load_chars,
        lora,
        trigger_words, photomake_mode, use_kolor, use_flux, make_dual_only, kolor_face, pulid, story_maker,
        input_id_emb_s_dict, input_id_img_s_dict, input_id_emb_un_dict, input_id_cloth_dict, guidance, condition_image,
        empty_emb_zero, use_cf, cf_scheduler, controlnet_path, controlnet_scale, cn_dict,input_tag_dict,SD35_mode,use_wrapper
):  # Corrected font_choice usage
    
    if len(general_prompt.splitlines()) >= 3:
        raise "Support for more than three characters is temporarily unavailable due to VRAM limitations, but this issue will be resolved soon."
    # _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    
    if not use_kolor and not use_flux and not story_maker:
        if model_type == "img2img" and "img" not in general_prompt:
            raise 'if using normal SDXL img2img ,need add the triger word " img "  behind the class word you want to customize, such as: man img or woman img'
    
    global total_length, attn_procs, cur_model_type
    global write
    global cur_step, attn_count
    
    # load_chars = load_character_files_on_running(unet, character_files=char_files)
    
    prompts_origin = prompt_array.splitlines()
    prompts_origin = [i.strip() for i in prompts_origin]
    prompts_origin = [i for i in prompts_origin if '[' in i]  # 删除空行
    # print(prompts_origin)
    prompts = [prompt for prompt in prompts_origin if not len(extract_content_from_brackets(prompt)) >= 2]  # 剔除双角色
    
    add_trigger_words = " " + trigger_words + " style "
    if lora:
        if lora in lora_lightning_list:
            prompts = remove_punctuation_from_strings(prompts)
        else:
            prompts = remove_punctuation_from_strings(prompts)
            prompts = [item + add_trigger_words for item in prompts]
    
    global character_index_dict, invert_character_index_dict, ref_indexs_dict, ref_totals
    global character_dict
    
    character_dict, character_list = character_to_dict(general_prompt, lora, add_trigger_words)
    # print(character_dict)
    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    # generator = torch.Generator(device=device).manual_seed(seed_)
    clipped_prompts = prompts[:]
    nc_indexs = []
    for ind, prompt in enumerate(clipped_prompts):
        if "[NC]" in prompt:
            nc_indexs.append(ind)
            if ind < id_length:
                raise f"The first {id_length} row is id prompts, cannot use [NC]!"
    
    prompts = [
        prompt if "[NC]" not in prompt else prompt.replace("[NC]", "")
        for prompt in clipped_prompts
    ]
    
    if lora:
        if lora in lora_lightning_list:
            prompts = [
                prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
            ]
        else:
            prompts = [
                prompt.rpartition("#")[0] + add_trigger_words if "#" in prompt else prompt for prompt in prompts
            ]
    else:
        prompts = [
            prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
        ]
    img_mode = False
    if kolor_face or pulid or (story_maker and not make_dual_only) or model_type == "img2img":
        img_mode = True
    # id_prompts = prompts[:id_length]
    (
        character_index_dict,
        invert_character_index_dict,
        replace_prompts,
        ref_indexs_dict,
        ref_totals,
    ) = process_original_prompt(character_dict, prompts, id_length, img_mode)
    
    if input_tag_dict:
        if len(input_tag_dict)<len(replace_prompts):
            raise "The number of input condition images is less than the number of scene prompts！"
        replace_prompts=[prompt +" " + input_tag_dict[i] for i,prompt in enumerate(replace_prompts)]
    #print(input_tag_dict)
    #print(replace_prompts)
    #[' a woman img, wearing a white T-shirt  wake up in the bed ;', ' a man img,wearing a suit,black hair.  is working.']
    # print(character_index_dict,invert_character_index_dict,replace_prompts,ref_indexs_dict,ref_totals)
    # character_index_dict：{'[Taylor]': [0, 3], '[sam]': [1, 2]},if 1 role {'[Taylor]': [0, 1, 2]}
    # invert_character_index_dict:{0: ['[Taylor]'], 1: ['[sam]'], 2: ['[sam]'], 3: ['[Taylor]']},if 1 role  {0: ['[Taylor]'], 1: ['[Taylor]'], 2: ['[Taylor]']}
    # ref_indexs_dict:{'[Taylor]': [0, 3], '[sam]': [1, 2]},if 1 role {'[Taylor]': [0]}
    # ref_totals: [0, 3, 1, 2]  if 1 role [0]
    if model_type == "img2img":
        # _upload_images = [_upload_images]
        input_id_images_dict = {}
        if len(upload_images) != len(character_dict.keys()):
            raise f"You upload images({len(upload_images)}) is not equal to the number of characters({len(character_dict.keys())})!"
        for ind, img in enumerate(upload_images):
            input_id_images_dict[character_list[ind]] = [img]  # 已经pil转化了 不用load {a:[img],b:[img]}
            # input_id_images_dict[character_list[ind]] = [load_image(img)]
    # real_prompts = prompts[id_length:]
    # if device == "cuda":
    #     torch.cuda.empty_cache()
    write = True
    cur_step = 0
    attn_count = 0
    total_results = []
    id_images = []
    results_dict = {}
    p_num = 0
    
    global cur_character
    if not load_chars:
        for character_key in character_dict.keys():  # 先生成角色对应第一句场景提示词的图片,图生图是批次生成
            character_key_str = character_key
            cur_character = [character_key]
            ref_indexs = ref_indexs_dict[character_key]
            current_prompts = [replace_prompts[ref_ind] for ref_ind in ref_indexs]
            if model_type == "txt2img":
                setup_seed(seed_)
            generator = torch.Generator(device=device).manual_seed(seed_)
            cur_step = 0
            cur_positive_prompts, cur_negative_prompt = apply_style(
                style_name, current_prompts, negative_prompt
            )
            print(f"Sampler  {character_key_str} 's cur_positive_prompts :{cur_positive_prompts}")
            if model_type == "txt2img":
                if use_flux:
                    if not use_cf:
                        id_images = pipe(
                            prompt=cur_positive_prompts,
                            num_inference_steps=_num_steps,
                            guidance_scale=guidance,
                            output_type="pil",
                            max_sequence_length=256,
                            height=height,
                            width=width,
                            generator=generator
                        ).images
                    else:
                        cur_negative_prompt = [cur_negative_prompt]
                        cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                            cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                        id_images = []
                        cfg = 1.0
                        for index, text in enumerate(cur_positive_prompts):
                            id_image = pipe.generate_image(
                                width=width,
                                height=height,
                                num_steps=_num_steps,
                                cfg=cfg,
                                guidance=guidance,
                                seed=seed_,
                                prompt=text,
                                negative_prompt=cur_negative_prompt[index],
                                cf_scheduler=cf_scheduler,
                                denoise=denoise_or_ip_sacle,
                                image=None
                            )
                            id_images.append(id_image)
                else:
                    if use_cf:
                        cur_negative_prompt = [cur_negative_prompt]
                        cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                            cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                        id_images = []
                        for index, text in enumerate(cur_positive_prompts):
                            id_image = pipe.generate_image(
                                width=width,
                                height=height,
                                num_steps=_num_steps,
                                cfg=cfg,
                                guidance=guidance,
                                seed=seed_,
                                prompt=text,
                                negative_prompt=cur_negative_prompt[index],
                                cf_scheduler=cf_scheduler,
                                denoise=denoise_or_ip_sacle,
                                image=None
                            )
                            id_images.append(id_image)
                    elif SD35_mode:
                        if use_wrapper:
                            id_images = pipe(
                                cur_positive_prompts,
                                num_inference_steps=_num_steps,
                                guidance_scale=guidance,
                                height=height,
                                width=width,
                                negative_prompt=cur_negative_prompt,
                                max_sequence_length=512,
                                generator=generator
                            )
                        else:
                            id_images = pipe(
                                cur_positive_prompts,
                                num_inference_steps=_num_steps,
                                guidance_scale=guidance,
                                height=height,
                                width=width,
                                negative_prompt=cur_negative_prompt,
                                max_sequence_length=512,
                                generator=generator
                            ).images
                    else:
                        if use_kolor:
                            cur_negative_prompt = [cur_negative_prompt]
                            cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                                cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                        id_images = pipe(
                            cur_positive_prompts,
                            num_inference_steps=_num_steps,
                            guidance_scale=cfg,
                            height=height,
                            width=width,
                            negative_prompt=cur_negative_prompt,
                            generator=generator
                        ).images
            
            elif model_type == "img2img":
                if use_kolor:
                    cur_negative_prompt = [cur_negative_prompt]
                    cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                        cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                    if kolor_face:
                        crop_image = input_id_img_s_dict[character_key_str][0]
                        face_embeds = input_id_emb_s_dict[character_key_str][0]
                        face_embeds = face_embeds.to(device, dtype=torch.float16)
                        if id_length > 1:
                            id_images = []
                            for index, i in enumerate(cur_positive_prompts):
                                id_image = pipe(
                                    prompt=i,
                                    negative_prompt=cur_negative_prompt[index],
                                    height=height,
                                    width=width,
                                    num_inference_steps=_num_steps,
                                    guidance_scale=cfg,
                                    num_images_per_prompt=1,
                                    generator=generator,
                                    face_crop_image=crop_image,
                                    face_insightface_embeds=face_embeds,
                                ).images
                                id_images.append(id_image)
                        else:
                            id_images = pipe(
                                prompt=cur_positive_prompts,
                                negative_prompt=cur_negative_prompt,
                                height=height,
                                width=width,
                                num_inference_steps=_num_steps,
                                guidance_scale=cfg,
                                num_images_per_prompt=1,
                                generator=generator,
                                face_crop_image=crop_image,
                                face_insightface_embeds=face_embeds,
                            ).images
                    else:
                        pipe.set_ip_adapter_scale([denoise_or_ip_sacle])
                        id_images = pipe(
                            prompt=cur_positive_prompts,
                            ip_adapter_image=input_id_images_dict[character_key],
                            negative_prompt=cur_negative_prompt,
                            num_inference_steps=_num_steps,
                            height=height,
                            width=width,
                            guidance_scale=cfg,
                            num_images_per_prompt=1,
                            generator=generator,
                        ).images
                elif use_flux:
                    if pulid:
                        id_embeddings = input_id_emb_s_dict[character_key_str][0]
                        uncond_id_embeddings = input_id_emb_un_dict[character_key_str][0]
                        if id_length > 1:
                            id_images = []
                            for index, i in enumerate(cur_positive_prompts):
                                id_image = pipe.generate_image(
                                    prompt=i,
                                    seed=seed_,
                                    start_step=2,
                                    num_steps=_num_steps,
                                    height=height,
                                    width=width,
                                    id_embeddings=id_embeddings,
                                    uncond_id_embeddings=uncond_id_embeddings,
                                    id_weight=1,
                                    guidance=guidance,
                                    true_cfg=1.0,
                                    max_sequence_length=128,
                                )
                                id_images.append(id_image)
                        
                        else:
                            id_images = pipe.generate_image(
                                prompt=cur_positive_prompts,
                                seed=seed_,
                                start_step=2,
                                num_steps=_num_steps,
                                height=height,
                                width=width,
                                id_embeddings=id_embeddings,
                                uncond_id_embeddings=uncond_id_embeddings,
                                id_weight=1,
                                guidance=guidance,
                                true_cfg=1.0,
                                max_sequence_length=128,
                            )
                            id_images = [id_images]
                    elif use_cf:
                        cfg = 1.0
                        cur_negative_prompt = [cur_negative_prompt]
                        cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                            cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                        id_images = []
                        for index, text in enumerate(cur_positive_prompts):
                            id_image = pipe.generate_image(
                                width=width,
                                height=height,
                                num_steps=_num_steps,
                                cfg=cfg,
                                guidance=guidance,
                                seed=seed_,
                                prompt=text,
                                negative_prompt=cur_negative_prompt[index],
                                cf_scheduler=cf_scheduler,
                                denoise=denoise_or_ip_sacle,
                                image=input_id_images_dict[character_key][0]
                            )
                            id_images.append(id_image)
                    else:
                        cfg = cfg if cfg <= 1 else cfg / 10 if 1 < cfg <= 10 else cfg / 100
                        id_images = pipe(
                            prompt=cur_positive_prompts,
                            image=input_id_images_dict[character_key],
                            strength=cfg,
                            latents=None,
                            num_inference_steps=_num_steps,
                            height=height,
                            width=width,
                            output_type="pil",
                            max_sequence_length=256,
                            guidance_scale=guidance,
                            generator=generator,
                        ).images
                
                elif story_maker and not make_dual_only:
                    img = input_id_images_dict[character_key][0]
                    # print(character_key_str,input_id_images_dict)
                    mask_image = input_id_img_s_dict[character_key_str][0]
                    face_info = input_id_emb_s_dict[character_key_str][0]
                    cloth_info = None
                    if isinstance(condition_image, torch.Tensor):
                        cloth_info = input_id_cloth_dict[character_key_str][0]
                    cur_negative_prompt = [cur_negative_prompt]
                    cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                        cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                    if id_length > 1:
                        id_images = []
                        for index, i in enumerate(cur_positive_prompts):
                            id_image = pipe(
                                image=img if not controlnet_path else [img, cn_dict[ref_indexs[index]][
                                    0]] if cn_dict else img,
                                mask_image=mask_image,
                                face_info=face_info,
                                prompt=i,
                                negative_prompt=cur_negative_prompt[index],
                                ip_adapter_scale=denoise_or_ip_sacle, lora_scale=0.8,
                                controlnet_conditioning_scale=controlnet_scale,
                                num_inference_steps=_num_steps,
                                guidance_scale=cfg,
                                height=height, width=width,
                                generator=generator,
                                cloth=cloth_info,
                            ).images
                            id_images.append(id_image)
                    else:
                        id_images = pipe(
                            image=img if not controlnet_path else [img, cn_dict[ref_indexs[0]][0]] if cn_dict else img,
                            mask_image=mask_image,
                            face_info=face_info,
                            prompt=cur_positive_prompts,
                            negative_prompt=cur_negative_prompt,
                            ip_adapter_scale=denoise_or_ip_sacle, lora_scale=0.8,
                            num_inference_steps=_num_steps,
                            guidance_scale=cfg,
                            height=height, width=width,
                            generator=generator,
                            cloth=cloth_info,
                        ).images
                
                else:
                    if use_cf:
                        cur_negative_prompt = [cur_negative_prompt]
                        cur_negative_prompt = cur_negative_prompt * len(cur_positive_prompts) if len(
                            cur_negative_prompt) != len(cur_positive_prompts) else cur_negative_prompt
                        id_images = []
                        for index, text in enumerate(cur_positive_prompts):
                            id_image = pipe.generate_image(
                                width=width,
                                height=height,
                                num_steps=_num_steps,
                                cfg=cfg,
                                guidance=guidance,
                                seed=seed_,
                                prompt=text,
                                negative_prompt=cur_negative_prompt[index],
                                cf_scheduler=cf_scheduler,
                                denoise=denoise_or_ip_sacle,
                                image=input_id_images_dict[character_key][0]
                            )
                            id_images.append(id_image)
                    elif SD35_mode:
                        if use_wrapper:
                            id_images = pipe(
                                cur_positive_prompts,
                                image=input_id_images_dict[character_key],
                                num_inference_steps=_num_steps,
                                guidance_scale=cfg,
                                strength=denoise_or_ip_sacle,
                                negative_prompt=cur_negative_prompt,
                                generator=generator,
                                max_sequence_length=512
                            )
                        else:
                            id_images = pipe(
                                cur_positive_prompts,
                                image=input_id_images_dict[character_key],
                                num_inference_steps=_num_steps,
                                guidance_scale=cfg,
                                strength=denoise_or_ip_sacle,
                                negative_prompt=cur_negative_prompt,
                                generator=generator,
                                max_sequence_length=512
                            ).images
                        
                    else:
                        if photomake_mode == "v2":
                            id_embeds = input_id_emb_s_dict[character_key_str][0]
                            id_images = pipe(
                                cur_positive_prompts,
                                input_id_images=input_id_images_dict[character_key],
                                num_inference_steps=_num_steps,
                                guidance_scale=cfg,
                                start_merge_step=start_merge_step,
                                height=height,
                                width=width,
                                negative_prompt=cur_negative_prompt,
                                id_embeds=id_embeds,
                                generator=generator
                            ).images
                        else:
                            # print("v1 mode,load_chars", cur_positive_prompts, negative_prompt,character_key )
                            id_images = pipe(
                                cur_positive_prompts,
                                input_id_images=input_id_images_dict[character_key],
                                num_inference_steps=_num_steps,
                                guidance_scale=cfg,
                                start_merge_step=start_merge_step,
                                height=height,
                                width=width,
                                negative_prompt=cur_negative_prompt,
                                generator=generator
                            ).images
            
            else:
                raise NotImplementedError(
                    "You should choice between original and Photomaker!",
                    f"But you choice {model_type}",
                )
            p_num += 1
            # total_results = id_images + total_results
            # yield total_results
            if story_maker and not make_dual_only and id_length > 1 and model_type == "img2img":
                for index, ind in enumerate(character_index_dict[character_key]):
                    results_dict[ref_totals[ind]] = id_images[index]
            elif pulid and id_length > 1 and model_type == "img2img":
                for index, ind in enumerate(character_index_dict[character_key]):
                    results_dict[ref_totals[ind]] = id_images[index]
            elif kolor_face and id_length > 1 and model_type == "img2img":
                for index, ind in enumerate(character_index_dict[character_key]):
                    results_dict[ref_totals[ind]] = id_images[index]
            elif use_flux and use_cf and id_length > 1 and model_type == "img2img":
                for index, ind in enumerate(character_index_dict[character_key]):
                    results_dict[ref_totals[ind]] = id_images[index]
            elif use_cf and not use_flux and id_length > 1 and model_type == "img2img":
                for index, ind in enumerate(character_index_dict[character_key]):
                    results_dict[ref_totals[ind]] = id_images[index]
            else:
                for ind, img in enumerate(id_images):
                    results_dict[ref_indexs[ind]] = img
            # real_images = []
            # print(results_dict)
            yield [results_dict[ind] for ind in results_dict.keys()]
    
    write = False
    if not load_chars:
        real_prompts_inds = [
            ind for ind in range(len(prompts)) if ind not in ref_totals
        ]
    else:
        real_prompts_inds = [ind for ind in range(len(prompts))]
    print(real_prompts_inds)
    real_prompt_no, negative_prompt_style = apply_style_positive(style_name, "real_prompt")
    negative_prompt = str(negative_prompt) + str(negative_prompt_style)
    # print(f"real_prompts_inds is {real_prompts_inds}")
    for real_prompts_ind in real_prompts_inds:  #
        real_prompt = replace_prompts[real_prompts_ind]
        cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        
        if model_type == "txt2img":
            setup_seed(seed_)
        generator = torch.Generator(device=device).manual_seed(seed_)
        
        if len(cur_character) > 1 and model_type == "img2img":
            raise "Temporarily Not Support Multiple character in Ref Image Mode!"
        cur_step = 0
        real_prompt, negative_prompt_style_no = apply_style_positive(style_name, real_prompt)
        print(f"Sample real_prompt : {real_prompt}")
        if model_type == "txt2img":
            # print(results_dict,real_prompts_ind)
            if use_flux:
                if not use_cf:
                    results_dict[real_prompts_ind] = pipe(
                        prompt=real_prompt,
                        num_inference_steps=_num_steps,
                        guidance_scale=guidance,
                        output_type="pil",
                        max_sequence_length=256,
                        height=height,
                        width=width,
                        generator=torch.Generator("cpu").manual_seed(seed_)
                    ).images[0]
                else:
                    cfg = 1.0
                    results_dict[real_prompts_ind] = pipe.generate_image(
                        width=width,
                        height=height,
                        num_steps=_num_steps,
                        cfg=cfg,
                        guidance=guidance,
                        seed=seed_,
                        prompt=real_prompt,
                        negative_prompt=negative_prompt,
                        cf_scheduler=cf_scheduler,
                        denoise=denoise_or_ip_sacle,
                        image=None
                    )
            else:
                if use_cf:
                    results_dict[real_prompts_ind] = pipe.generate_image(
                        width=width,
                        height=height,
                        num_steps=_num_steps,
                        cfg=cfg,
                        guidance=guidance,
                        seed=seed_,
                        prompt=real_prompt,
                        negative_prompt=negative_prompt,
                        cf_scheduler=cf_scheduler,
                        denoise=denoise_or_ip_sacle,
                        image=None
                    )
                elif SD35_mode:
                    if use_wrapper:
                        results_dict[real_prompts_ind] = pipe(
                            real_prompt,
                            num_inference_steps=_num_steps,
                            guidance_scale=cfg,
                            height=height,
                            width=width,
                            negative_prompt=negative_prompt,
                            generator=generator,
                            max_sequence_length=512,
                        )
                    else:
                        results_dict[real_prompts_ind] = pipe(
                            real_prompt,
                            num_inference_steps=_num_steps,
                            guidance_scale=cfg,
                            height=height,
                            width=width,
                            negative_prompt=negative_prompt,
                            generator=generator,
                            max_sequence_length=512,
                        ).images[0]
                    
                else:
                    results_dict[real_prompts_ind] = pipe(
                        real_prompt,
                        num_inference_steps=_num_steps,
                        guidance_scale=cfg,
                        height=height,
                        width=width,
                        negative_prompt=negative_prompt,
                        generator=generator,
                    ).images[0]
        
        elif model_type == "img2img":
            empty_img = Image.new('RGB', (height, width), (255, 255, 255))
            if use_kolor:
                if kolor_face:
                    empty_image = Image.new('RGB', (336, 336), (255, 255, 255))
                    crop_image = input_id_img_s_dict[
                        cur_character[0]] if real_prompts_ind not in nc_indexs else empty_image
                    face_embeds = input_id_emb_s_dict[cur_character[0]][
                        0] if real_prompts_ind not in nc_indexs else empty_emb_zero
                    face_embeds = face_embeds.to(device, dtype=torch.float16)
                    results_dict[real_prompts_ind] = pipe(
                        prompt=real_prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=_num_steps,
                        guidance_scale=cfg,
                        num_images_per_prompt=1,
                        generator=generator,
                        face_crop_image=crop_image,
                        face_insightface_embeds=face_embeds,
                    ).images[0]
                else:
                    results_dict[real_prompts_ind] = pipe(
                        prompt=real_prompt,
                        ip_adapter_image=(
                            input_id_images_dict[cur_character[0]]
                            if real_prompts_ind not in nc_indexs
                            else empty_img
                        ),
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=_num_steps,
                        guidance_scale=cfg,
                        num_images_per_prompt=1,
                        generator=generator,
                        nc_flag=True if real_prompts_ind in nc_indexs else False,  # nc_flag，用索引标记，主要控制非角色人物的生成，默认false
                    ).images[0]
            elif use_flux:
                if pulid:
                    id_embeddings = input_id_emb_s_dict[cur_character[0]][
                        0] if real_prompts_ind not in nc_indexs else empty_emb_zero
                    uncond_id_embeddings = input_id_emb_un_dict[cur_character[0]][
                        0] if real_prompts_ind not in nc_indexs else empty_emb_zero
                    results_dict[real_prompts_ind] = pipe.generate_image(
                        prompt=real_prompt,
                        seed=seed_,
                        start_step=2,
                        num_steps=_num_steps,
                        height=height,
                        width=width,
                        id_embeddings=id_embeddings,
                        uncond_id_embeddings=uncond_id_embeddings,
                        id_weight=1,
                        guidance=guidance,
                        true_cfg=1.0,
                        max_sequence_length=128,
                    )
                elif use_cf:
                    cfg = 1.0
                    results_dict[real_prompts_ind] = pipe.generate_image(
                        width=width,
                        height=height,
                        num_steps=_num_steps,
                        cfg=cfg,
                        guidance=guidance,
                        seed=seed_,
                        prompt=real_prompt,
                        negative_prompt=negative_prompt,
                        cf_scheduler=cf_scheduler,
                        denoise=denoise_or_ip_sacle,
                        image=(input_id_images_dict[cur_character[0]]
                               if real_prompts_ind not in nc_indexs
                               else empty_img
                               )
                    )
                else:
                    cfg = cfg if cfg <= 1 else cfg / 10 if 1 < cfg <= 10 else cfg / 100
                    results_dict[real_prompts_ind] = pipe(
                        prompt=real_prompt,
                        image=(
                            input_id_images_dict[cur_character[0]]
                            if real_prompts_ind not in nc_indexs
                            else empty_img
                        ),
                        latents=None,
                        strength=cfg,
                        num_inference_steps=_num_steps,
                        height=height,
                        width=width,
                        output_type="pil",
                        max_sequence_length=256,
                        guidance_scale=guidance,
                        generator=generator,
                    ).images[0]
            elif story_maker and not make_dual_only:
                mask_image = input_id_img_s_dict[cur_character[0]][0]
                img_2 = input_id_images_dict[cur_character[0]][0] if real_prompts_ind not in nc_indexs else empty_img
                cloth_info = None
                if isinstance(condition_image, torch.Tensor):
                    if controlnet_path:
                        cn_img = input_id_cloth_dict[cur_character[0]][0]
                        img_2 = [img_2, cn_img]
                    else:
                        cloth_info = input_id_cloth_dict[cur_character[0]][0]
                face_info = input_id_emb_s_dict[cur_character[0]][
                    0] if real_prompts_ind not in nc_indexs else empty_emb_zero
                
                results_dict[real_prompts_ind] = pipe(
                    image=img_2 if not controlnet_path else [img_2, cn_dict[real_prompts_ind][0]] if cn_dict else img_2,
                    mask_image=mask_image,
                    face_info=face_info,
                    prompt=real_prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_scale=denoise_or_ip_sacle, lora_scale=0.8,
                    num_inference_steps=_num_steps,
                    guidance_scale=cfg,
                    controlnet_conditioning_scale=controlnet_scale,
                    height=height, width=width,
                    generator=generator,
                    cloth=cloth_info,
                ).images[0]
            else:
                if use_cf:
                    results_dict[real_prompts_ind] = pipe.generate_image(
                        width=width,
                        height=height,
                        num_steps=_num_steps,
                        cfg=cfg,
                        guidance=guidance,
                        seed=seed_,
                        prompt=real_prompt,
                        negative_prompt=negative_prompt,
                        cf_scheduler=cf_scheduler,
                        denoise=denoise_or_ip_sacle,
                        image=(input_id_images_dict[cur_character[0]]
                               if real_prompts_ind not in nc_indexs
                               else empty_img
                               )
                    )
                elif SD35_mode:
                    # print(real_prompts_ind, real_prompt, "v1 mode", )
                    if use_wrapper:
                        results_dict[real_prompts_ind] = pipe(
                            real_prompt,
                            image=(
                                input_id_images_dict[cur_character[0]]
                                if real_prompts_ind not in nc_indexs
                                else input_id_images_dict[character_list[0]]
                            ),
                            num_inference_steps=_num_steps,
                            strength=denoise_or_ip_sacle,
                            guidance_scale=cfg,
                            negative_prompt=negative_prompt,
                            generator=generator,
                            max_sequence_length=512,
                        )
                    else:
                        results_dict[real_prompts_ind] = pipe(
                            real_prompt,
                            image=(
                                input_id_images_dict[cur_character[0]]
                                if real_prompts_ind not in nc_indexs
                                else input_id_images_dict[character_list[0]]
                            ),
                            num_inference_steps=_num_steps,
                            strength=denoise_or_ip_sacle,
                            guidance_scale=cfg,
                            negative_prompt=negative_prompt,
                            generator=generator,
                            max_sequence_length=512,
                        ).images[0]
                   
                else:
                    if photomake_mode == "v2":
                        # V2版本必须要有id_embeds，只能用input_id_images作为风格参考
                        print(cur_character)
                        id_embeds = input_id_emb_s_dict[cur_character[0]][
                            0] if real_prompts_ind not in nc_indexs else empty_emb_zero
                        results_dict[real_prompts_ind] = pipe(
                            real_prompt,
                            input_id_images=(
                                input_id_images_dict[cur_character[0]]
                                if real_prompts_ind not in nc_indexs
                                else input_id_images_dict[character_list[0]]
                            ),
                            num_inference_steps=_num_steps,
                            guidance_scale=cfg,
                            start_merge_step=start_merge_step,
                            height=height,
                            width=width,
                            negative_prompt=negative_prompt,
                            generator=generator,
                            id_embeds=id_embeds,
                            nc_flag=True if real_prompts_ind in nc_indexs else False,
                        ).images[0]
                    else:
                        # print(real_prompts_ind, real_prompt, "v1 mode", )
                        results_dict[real_prompts_ind] = pipe(
                            real_prompt,
                            input_id_images=(
                                input_id_images_dict[cur_character[0]]
                                if real_prompts_ind not in nc_indexs
                                else input_id_images_dict[character_list[0]]
                            ),
                            num_inference_steps=_num_steps,
                            guidance_scale=cfg,
                            start_merge_step=start_merge_step,
                            height=height,
                            width=width,
                            negative_prompt=negative_prompt,
                            generator=generator,
                            nc_flag=True if real_prompts_ind in nc_indexs else False,
                            # nc_flag，用索引标记，主要控制非角色人物的生成，默认false
                        ).images[0]
        
        else:
            raise NotImplementedError(
                "You should choice between original and Photomaker!",
                f"But you choice {model_type}",
            )
        yield [results_dict[ind] for ind in results_dict.keys()]
    total_results = [results_dict[ind] for ind in range(len(prompts))]
    torch.cuda.empty_cache()
    yield total_results


class Storydiffusion_Model_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_prompt": ("STRING", {"multiline": True,
                                                "default": "[Taylor] a woman img, wearing a white T-shirt, blue loose hair.\n"
                                                           "[Lecun] a man img,wearing a suit,black hair."}),
                "repo_id": ("STRING", {"default": ""}),
                "ckpt_name": (["none"]+folder_paths.get_filename_list("checkpoints"),),
                "vae_id":(["none"]+folder_paths.get_filename_list("vae"),),
                "character_weights": (["none"]+find_directories(base_pt),),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "controlnet_model": (["none"] + folder_paths.get_filename_list("controlnet"),),
                "clip_vision": (["none"] + folder_paths.get_filename_list("clip_vision"),),
                "trigger_words": ("STRING", {"default": "best quality"}),
                "sampeler_name": (SAMPLER_NAMES,),
                "scheduler": (SCHEDULER_NAMES,),
                "sa32_degree": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "sa64_degree": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32, "display": "number"}),
                "photomake_mode": (["v1", "v2"],),
                "easy_function":("STRING", {"default": ""}),
            },
            "optional": {"image": ("IMAGE",),
                         "condition_image": ("IMAGE",),
                         "model": ("MODEL",),
                         "clip":("CLIP",),
                         "vae":("VAE",),
        },
        }

    RETURN_TYPES = ("STORY_DICT", )
    RETURN_NAMES = ("model", )
    FUNCTION = "story_model_loader"
    CATEGORY = "Storydiffusion"
    
    def story_model_loader(self,character_prompt, repo_id, ckpt_name, vae_id, character_weights, lora, lora_scale,controlnet_model, clip_vision,trigger_words,sampeler_name, scheduler,
                           sa32_degree, sa64_degree, width, height, photomake_mode, easy_function,**kwargs):
        
        cf_model=kwargs.get("model")
        clip = kwargs.get("clip")
        front_vae=kwargs.get("vae")
        
        scheduler_choice = get_scheduler(sampeler_name,scheduler)
        scheduler={"name":sampeler_name,"scheduler":scheduler}
        id_number=len(character_prompt.splitlines())
        if id_number > 2:
            id_number=2
        
        image = kwargs.get("image")
        if isinstance(image,torch.Tensor):
            #print(image.shape)
            batch_num,_,_,_=image.size()
            model_type="img2img"
            if batch_num!=id_number:
                raise "role prompt numbers don't match input image numbers...example:2 roles need 2 input images,"
        else:
            model_type = "txt2img"
            image=None
            
        logging.info(f"Process using {id_number} roles,mode is {model_type}....")
        
        if controlnet_model=="none":
            controlnet_path=None
        else:
            controlnet_path=folder_paths.get_full_path("controlnet", controlnet_model)

        condition_image=kwargs.get("condition_image")
        photomaker_path = os.path.join(photomaker_dir, f"photomaker-{photomake_mode}.bin")
        photomake_mode_=photomake_mode
       
        # load model
        (auraface, NF4, save_model, kolor_face,flux_pulid_name,pulid,quantized_mode,story_maker,make_dual_only,
         clip_vision_path,char_files,ckpt_path,lora,lora_path,use_kolor,photomake_mode,use_flux,onnx_provider,low_vram,TAG_mode,SD35_mode,consistory,cached,inject,use_quantize)=get_easy_function(
            easy_function,clip_vision,character_weights,ckpt_name,lora,repo_id,photomake_mode)
        
        
        photomaker_path,face_ckpt,photomake_mode,pulid_ckpt,face_adapter,kolor_ip_path=pre_checkpoint(
            photomaker_path, photomake_mode, kolor_face, pulid, story_maker, clip_vision_path,use_kolor,model_type)
    
        if total_vram > 45000.0:
            aggressive_offload = False
            offload = False
        elif 17000.0<total_vram < 45000.0:
            aggressive_offload = False
            offload = True
        else:
            aggressive_offload = True
            offload = True
        logging.info(f"total_vram is {total_vram},aggressive_offload is {aggressive_offload},offload is {offload}")
        # global
        global  attn_procs
        global sa32, sa64, write, height_s, width_s
        global attn_count, total_count, id_length, total_length, cur_step

        sa32 = sa32_degree
        sa64 = sa64_degree
        attn_count = 0
        total_count = 0
        cur_step = 0
        id_length = id_number
        total_length = 5
        attn_procs = {}
        write = False
        height_s = height
        width_s = width
        use_cf=False
        use_storydif=False
        use_wrapper = False
        if not repo_id and not ckpt_path and not cf_model:
            raise "you need choice a model or repo_id or a comfyUI model..."
        elif not repo_id and not ckpt_path and cf_model:
            from comfy.utils import load_torch_file as load_torch_file_
            from comfy.sd import VAE as cf_vae
            from .utils.comfy_normal import CFGenerator
            if vae_id != "none":
                vae_path = folder_paths.get_full_path("vae", vae_id)
                sd = load_torch_file_(vae_path)
                vae = cf_vae(sd=sd)
            else:
                if not front_vae:
                    raise "Now,using comfyUI normal processing  must choice 'vae' or 'ae' from  'vae' menu."
                else:
                    vae = front_vae
            if not clip:
                raise "Use comfyUI normal processing must need comfyUI clip,if using flux need dual clip ."
            if consistory：
                raise "Use comfyUI calss processing don't support consistory mode,please checke readme how to use consistory mode."
            use_cf = True
            if cf_model.model.model_type.value==8:
                use_flux=True
                cf_model_type="FLUX"
            elif cf_model.model.model_type.value==4:
                cf_model_type = "CASCADE"
            elif cf_model.model.model_type.value==6:
                cf_model_type = "FLOW"
            elif cf_model.model.model_type.value==5:
                cf_model_type = "EDM"
            elif cf_model.model.model_type.value==1:
                cf_model_type = "EPS"
            elif cf_model.model.model_type.value == 2:
                cf_model_type = "V_PREDICTION"
            elif cf_model.model.model_type.value == 3:
                cf_model_type = "V_PREDICTION_EDM"
            elif cf_model.model.model_type.value == 7:
                cf_model_type = "PREDICTION_CONTINUOUS"
            else:
                try:
                    use_flux = True
                    cf_model_type = "FLUX"
                except:
                    raise "unsupport checkpoints"
            pipe=CFGenerator(cf_model,clip,vae,cf_model_type,device)
            pulid = False
        elif not repo_id and ckpt_path: # load ckpt
            if_repo = False
            if story_maker:
                if not make_dual_only: #default dual
                    logging.info("start story-make processing...")
                    pipe=story_maker_loader(clip_load,clip_vision_path,dir_path,ckpt_path, face_adapter,UniPCMultistepScheduler,controlnet_path,lora_scale,low_vram)
                else:
                    use_storydif=True
                    logging.info("start story-diffusion and story-make processing...")
                    pipe = load_models(ckpt_path, model_type=model_type, single_files=True, use_safetensors=True,
                                       photomake_mode=photomake_mode, photomaker_path=photomaker_path, lora=lora,
                                       lora_path=lora_path,
                                       trigger_words=trigger_words, lora_scale=lora_scale)
                    set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            elif "flux" in ckpt_path.lower() or use_flux:
                use_flux=True
                if pulid:
                    logging.info("start flux-pulid processing...")
                    from .PuLID.app_flux import FluxGenerator
                    if not clip_vision_path:
                        raise "need 'EVA02_CLIP_L_336_psz14_s6B.pt' in comfyUI/models/clip_vision"
                    if NF4:
                        quantized_mode = "nf4"
                    if vae_id == "none" :
                        if not front_vae:
                           raise "Now,using pulid must choice 'ae' from 'vae' menu."
                        else:
                           raise "Now,using pulid must choice 'ae' from 'vae' menu(come soon)"
                    else:
                        vae_path = folder_paths.get_full_path("vae", vae_id)
                    pipe = FluxGenerator(flux_pulid_name, ckpt_path, "cuda", offload=offload,
                                         aggressive_offload=aggressive_offload, pretrained_model=pulid_ckpt,
                                         quantized_mode=quantized_mode, clip_vision_path=clip_vision_path, clip_cf=clip,
                                         vae_cf=vae_path,if_repo=if_repo,onnx_provider=onnx_provider,use_quantize=use_quantize)
                else:
                    raise "flux don't support single checkpoints loading now"
            
            elif "3.5" in ckpt_path.lower() and clip and (vae_id!="none" or front_vae):
                logging.info("start sd3.5 mode processing...")
                sd35repo = os.path.join(dir_path, "config/stable-diffusion-3.5-large")
                vae_config=os.path.join(sd35repo,"vae")
                cf_vae = False
                if vae_id!="none":
                    vae_path = folder_paths.get_full_path("vae", vae_id)
                    vae = AutoencoderKL.from_single_file(vae_path,config=vae_config, torch_dtype=torch.bfloat16)
                    #vae = AutoencoderKL.from_pretrained("F:/test/ComfyUI/models/diffusers/stabilityai/stable-diffusion-3.5-large/vae", torch_dtype=torch.bfloat16)
                else:
                    if front_vae:
                        vae=front_vae
                        cf_vae = True
                    else:
                        raise "need choice a vae model,or link vae in the front."
                pipe=SD35Wrapper(ckpt_path,clip,vae,cf_vae,sd35repo,dir_path)
                pipe.pipe.enable_model_cpu_offload()
                use_wrapper=True
                use_storydif = False
            elif consistory:
                logging.info("start consistory mode processing...")
                from .consistory.consistory_run import load_pipeline
                pipe=load_pipeline(repo_id,ckpt_path,gpu_id=0)
                if lora is not None:
                    active_lora = pipe.get_active_adapters()
                    #print(active_lora)
                    if active_lora:
                        pipe.unload_lora_weights()  # make sure lora is not mix
                    if lora in lora_lightning_list:
                        pipe.load_lora_weights(lora_path)
                    else:
                        pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                use_storydif = False
            else:
                logging.info("start story-diffusion mode processing...")
                use_storydif=True
                pipe = load_models(ckpt_path, model_type=model_type, single_files=True, use_safetensors=True,
                                   photomake_mode=photomake_mode, photomaker_path=photomaker_path, lora=lora,
                                   lora_path=lora_path,
                                   trigger_words=trigger_words, lora_scale=lora_scale)
                set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
                
        else: #if repo or no ckpt,choice repo
            if_repo=True
            if repo_id.rsplit("/")[-1].lower()=="playground-v2.5-1024px-aesthetic":
                logging.info("start playground story-diffusion  processing...")
                use_storydif = True
                pipe = DiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float16,
                )
                set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            elif repo_id.rsplit("/")[-1].lower()=="sdxl-unstable-diffusers-y":
                logging.info("start sdxl-unstable story-diffusion  processing...")
                use_storydif = True
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    repo_id, torch_dtype=torch.float16,use_safetensors=False
                )
                set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            elif use_kolor:
                logging.info("start kolor processing...")
                if transformers_v>=4.45:
                    import shutil
                    print("transformers_v>=4.45 cause error,try fix it")
                    chatglm_config_fix = os.path.join(dir_path, "kolors", "tokenization_chatglm.py")
                    chatglm_config_origin = os.path.join(repo_id, "text_encoder", "tokenization_chatglm.py")
                    chatglm_config_origin_tokens = os.path.join(repo_id, "tokenizer", "tokenization_chatglm.py")
                    try:
                        if os.path.exists(chatglm_config_fix):
                            shutil.copy2(chatglm_config_fix, chatglm_config_origin)
                            shutil.copy2(chatglm_config_fix, chatglm_config_origin_tokens)
                            print(f"replace {chatglm_config_origin_tokens}  and{chatglm_config_origin} from {chatglm_config_fix}")
                    except:
                        print(f"fix fail,you can copy from {chatglm_config_fix} ,then cover {chatglm_config_origin} and {chatglm_config_origin_tokens}")
                        
                pipe=kolor_loader(repo_id, model_type, set_attention_processor, id_length, kolor_face, clip_vision_path,
                             clip_load, CLIPVisionModelWithProjection, CLIPImageProcessor,
                             photomaker_dir, face_ckpt, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel)
                pipe.enable_model_cpu_offload()
                use_storydif = False
            elif use_flux:
                from .model_loader_utils import flux_loader
                pipe=flux_loader(folder_paths,ckpt_path,repo_id,AutoencoderKL,save_model,model_type,pulid,clip_vision_path,NF4,vae_id,offload,aggressive_offload,pulid_ckpt,quantized_mode,
                if_repo,dir_path,clip,onnx_provider,use_quantize)
                if lora:
                    if not "Hyper" in lora_path : #can't support Hyper now
                        if not NF4:
                            logging.info("try using lora in flux quantize processing...")
                            pipe.load_lora_weights(lora_path)
                            pipe.fuse_lora(lora_scale=0.125)  # lora_scale=0.125
            elif SD35_mode:
                logging.info("start sd3.5 processing...")
                pipe=sd35_loader(repo_id,ckpt_path,dir_path,NF4,model_type,lora, lora_path, lora_scale,)
                pipe.enable_model_cpu_offload()
                use_storydif = False
            elif consistory:
                logging.info("start consistory mode processing...")
                from .consistory.consistory_run import load_pipeline
                pipe = load_pipeline(repo_id, ckpt_path,gpu_id=0)
                if lora is not None:
                    active_lora = pipe.get_active_adapters()
                    if active_lora:
                        pipe.unload_lora_weights()  # make sure lora is not mix
                    if lora in lora_lightning_list:
                        pipe.load_lora_weights(lora_path)
                    else:
                        pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                
            else: # SDXL dif_repo
                if  story_maker:
                    if not make_dual_only:
                        logging.info("start story_maker processing...")
                        from .StoryMaker.pipeline_sdxl_storymaker import StableDiffusionXLStoryMakerPipeline
                        
                        pipe = StableDiffusionXLStoryMakerPipeline.from_pretrained(
                            repo_id, torch_dtype=torch.float16)
                        controlnet=None
                        if controlnet_path:
                            from diffusers import ControlNetModel
                            from safetensors.torch import load_file
                            controlnet = ControlNetModel.from_unet(pipe.unet)
                            cn_state_dict = load_file(controlnet_path, device="cpu")
                            controlnet.load_state_dict(cn_state_dict, strict=False)
                            del cn_state_dict
                            controlnet.to(torch.float16)
                        if device != "mps":
                            if not low_vram:
                                pipe.cuda()
                        image_encoder=clip_load(clip_vision_path)
                        pipe.load_storymaker_adapter(image_encoder, face_adapter, scale=0.8, lora_scale=lora_scale,controlnet=controlnet)
                        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                        #pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
                        #pipe.enable_vae_slicing()
                        if device != "mps":
                            if low_vram:
                                pipe.enable_model_cpu_offload()
                else:
                    logging.info("start story_diffusion processing...")
                    use_storydif = True
                    pipe = load_models(repo_id, model_type=model_type, single_files=False, use_safetensors=True,
                                       photomake_mode=photomake_mode,
                                       photomaker_path=photomaker_path, lora=lora,
                                       lora_path=lora_path,
                                       trigger_words=trigger_words, lora_scale=lora_scale)
                    set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
                    
        if vae_id != "none":
            vae_id = folder_paths.get_full_path("vae", vae_id)
            vae_config = os.path.join(dir_path, "local_repo", "vae")
            if use_storydif:
                pipe.vae=AutoencoderKL.from_single_file(vae_id, config=vae_config,torch_dtype=torch.float16)
            elif consistory:
                pipe.vae = AutoencoderKL.from_single_file(vae_id, config=vae_config,torch_dtype=torch.float16)
        load_chars = False
        if use_storydif:
            pipe.scheduler = scheduler_choice.from_config(pipe.scheduler.config)
            load_chars = load_character_files_on_running(pipe.unet, character_files=char_files)
            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            pipe.enable_vae_slicing()
            if device != "mps":
                if low_vram:
                    pipe.enable_model_cpu_offload()
        
        torch.cuda.empty_cache()
        # need get emb
        character_name_dict_, character_list_ = character_to_dict(character_prompt, lora, trigger_words)
        #print(character_list_)
        if model_type=="img2img":
            d1, _, _, _ = image.size()
            if d1 == 1:
                image_load = [nomarl_upscale(image, width, height)]
            else:
                img_list = list(torch.chunk(image, chunks=d1))
                image_load = [nomarl_upscale(img, width, height) for img in img_list]
                
            from .model_loader_utils import insight_face_loader,get_insight_dict
            app_face,pipeline_mask,app_face_=insight_face_loader(photomake_mode, auraface, kolor_face, story_maker, make_dual_only,use_storydif)
            input_id_emb_s_dict, input_id_img_s_dict, input_id_emb_un_dict, input_id_cloth_dict=get_insight_dict(app_face,pipeline_mask,app_face_,image_load,photomake_mode,
                                                                                                                 kolor_face,story_maker,make_dual_only,
                     pulid,pipe,character_list_,condition_image,width, height,use_storydif)
        else:
            input_id_emb_s_dict = {}
            input_id_img_s_dict = {}
            input_id_emb_un_dict ={}
            input_id_cloth_dict = {}
        #print(input_id_img_s_dict)
        role_name_list = [i for i in character_name_dict_.keys()]
        
        if TAG_mode and isinstance(condition_image,torch.Tensor) and cf_model: #using cf_model as tag input
            k1, _, _, _ = condition_image.size()
            if k1 == 1:
                image_tag = [nomarl_upscale(condition_image, width, height)]
            else:
                img_list = list(torch.chunk(condition_image, chunks=k1))
                image_tag = [nomarl_upscale(img, width, height) for img in img_list]
            input_tag_dict = {}
            for i,img in enumerate(image_tag):
                input_tag_dict[i] =cf_model.run_tag(img)
            del cf_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            input_tag_dict={}
        
        #print( role_name_list)
        model={"pipe":pipe,"use_flux":use_flux,"use_kolor":use_kolor,"photomake_mode":photomake_mode,"trigger_words":trigger_words,"lora_scale":lora_scale,
               "load_chars":load_chars,"repo_id":repo_id,"lora_path":lora_path,"ckpt_path":ckpt_path,"model_type":model_type, "lora": lora,
               "scheduler":scheduler,"width":width,"height":height,"kolor_face":kolor_face,"pulid":pulid,"story_maker":story_maker,
               "make_dual_only":make_dual_only,"face_adapter":face_adapter,"clip_vision_path":clip_vision_path,"consistory":consistory,"cached":cached,"inject":inject,
               "controlnet_path":controlnet_path,"character_prompt":character_prompt,"image":image,"condition_image":condition_image,
               "input_id_emb_s_dict":input_id_emb_s_dict,"input_id_img_s_dict":input_id_img_s_dict,"use_cf":use_cf,"SD35_mode":SD35_mode,"use_wrapper":use_wrapper,
               "input_id_emb_un_dict":input_id_emb_un_dict,"input_id_cloth_dict":input_id_cloth_dict,"role_name_list":role_name_list,"use_storydif":use_storydif,"low_vram":low_vram,"input_tag_dict":input_tag_dict}
        return (model,)


class Storydiffusion_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STORY_DICT",),
                "scene_prompts": ("STRING", {"multiline": True,
                                             "default": "[Taylor] wake up in the bed ;\n[Taylor] have breakfast by the window;\n[Lecun] driving a car;\n[Lecun] is working."}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "bad anatomy, bad hands, missing fingers, extra fingers, "
                                                          "three hands, three legs, bad arms, missing legs, "
                                                          "missing arms, poorly drawn face, bad face, fused face, "
                                                          "cloned face, three crus, fused feet, fused thigh, "
                                                          "extra crus, ugly fingers, horn,"
                                                          "amputation, disconnected limbs"}),
                "img_style": (
                    ["No_style", "Realistic", "Japanese_Anime", "Digital_Oil_Painting", "Pixar_Disney_Character",
                     "Photographic", "Comic_book",
                     "Line_art", "Black_and_White_Film_Noir", "Isometric_Rooms"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "denoise_or_ip_sacle": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "guidance": (
                    "FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "mask_threshold": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "start_step": ("INT", {"default": 5, "min": 1, "max": 1024}),
                "save_character": ("BOOLEAN", {"default": False},),
                "controlnet_scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "guidance_list": ("STRING", {"multiline": True, "default": "0., 0.25, 0.4, 0.75;0.6, 0.25, 1., 0.75"}),
            },
            "optional": {"control_image": ("IMAGE",),
                         },
            }

      
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "prompt_array",)
    FUNCTION = "story_sampler"
    CATEGORY = "Storydiffusion"

    def story_sampler(self, model,scene_prompts, negative_prompt, img_style, seed, steps,
                  cfg, denoise_or_ip_sacle, style_strength_ratio,
                  guidance, mask_threshold, start_step,save_character,controlnet_scale,guidance_list,**kwargs):
        # get value from dict
        pipe=model.get("pipe")
        use_flux=model.get("use_flux")
        photomake_mode=model.get("photomake_mode")
        use_kolor=model.get("use_kolor")
        load_chars=model.get("load_chars")
        model_type=model.get("model_type")
        trigger_words=model.get("trigger_words")
        repo_id=model.get("repo_id")
        lora_path=model.get("lora_path")
        ckpt_path=model.get("ckpt_path")
        lora=model.get("lora")
        lora_scale =model.get("lora_scale")
        scheduler=model.get("scheduler")
        height=model.get("height")
        width = model.get("width")
        kolor_face= model.get("kolor_face")
        story_maker=model.get("story_maker")
        face_adapter=model.get("face_adapter")
        pulid= model.get("pulid")
        controlnet_path=model.get("controlnet_path")
        clip_vision_path=model.get("clip_vision_path")
        make_dual_only= model.get("make_dual_only")
        scheduler_choice = get_scheduler(scheduler["name"],scheduler["scheduler"])
        character_prompt=model.get("character_prompt")
        condition_image=model.get("condition_image")
        image=model.get("image")
        use_cf=model.get("use_cf")
        input_id_emb_s_dict = model.get("input_id_emb_s_dict")
        input_id_img_s_dict = model.get("input_id_img_s_dict")
        input_id_emb_un_dict = model.get("input_id_emb_un_dict")
        input_id_cloth_dict = model.get("input_id_cloth_dict")
        role_name_list=model.get("role_name_list")
        use_storydif=model.get("use_storydif")
        low_vram=model.get("low_vram")
        SD35_mode=model.get("SD35_mode")
        cf_scheduler=scheduler
        #print(input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,role_name_list) #'[Taylor]',['[Taylor]']
        control_image=kwargs.get("control_image")
        input_tag_dict=model.get("input_tag_dict")
        use_wrapper=model.get("use_wrapper")
        consistory=model.get("consistory")
        cached=model.get("cached")
        inject=model.get("inject")
        if use_storydif:
            pipe.to(device)

        empty_emb_zero = None
        if model_type=="img2img":           
            if pulid or kolor_face or (photomake_mode=="v2" and use_storydif):
                empty_emb_zero = torch.zeros_like(input_id_emb_s_dict[role_name_list[0]][0]).to(device)
        
        # 格式化文字内容
        scene_prompts=scene_prompts.strip()
        character_prompt=character_prompt.strip()
        # 从角色列表获取角色方括号信息
        char_origin = character_prompt.splitlines()
        char_origin=[i for i in char_origin if "[" in i]
        #print(char_origin)
        char_describe = char_origin # [A a men...,B a girl ]
        char_origin = ["["+ char.split("]")[0].split("[")[-1] +"]" for char in char_origin]
        #print(char_origin)
        # 判断是否有双角色prompt，如果有，获取双角色列表及对应的位置列表，
        prompts_origin = scene_prompts.splitlines()
        prompts_origin=[i.strip() for i in prompts_origin]
        prompts_origin = [i for i in prompts_origin if "[" in i]
        #print(prompts_origin)
        positions_dual = [index for index, prompt in enumerate(prompts_origin) if len(extract_content_from_brackets(prompt))>=2]  #改成单句中双方括号方法，利于MS组句，[A]... [B]...[C]
        prompts_dual = [prompt for prompt in prompts_origin if len(extract_content_from_brackets(prompt))>=2]
        
        positions_nc = [index for index, prompt in enumerate(prompts_origin) if"[NC]" in prompt]  # 找到NC的位置
        prompts_no_dual = [prompt for prompt in prompts_origin if not len(extract_content_from_brackets(prompt)) >= 2]
        prompts_no_nc_dual = [prompt for prompt in prompts_no_dual if "[NC]" not in prompt]
        
        if len(char_origin) == 2:
            positions_char_1 = [index for index, prompt in enumerate(prompts_origin) if char_origin[0] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置
            positions_char_2 = [index for index, prompt in enumerate(prompts_origin) if char_origin[1] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置
        
        cn_dict = {}  # {0:[img]} story_maker
        if isinstance(control_image, torch.Tensor) and story_maker and controlnet_path and not make_dual_only:
            f1, _, _, _ = control_image.size()
            if f1!=len(prompts_origin):
                raise "if using story-maker controlnet,The number of input control-images and scene prompts should be consistent!! "
            else:
                if f1 == 1:
                    cn_image_load = [nomarl_upscale(control_image, width, height)]
                else:
                    img_list = list(torch.chunk(control_image, chunks=f1))
                    cn_image_load = [nomarl_upscale(img, width, height) for img in img_list]
                    
                for index, img in enumerate(cn_image_load):
                    cn_dict[index] = [img]
                
                for i in  positions_dual:  #  dual to {dual_num:[None]}
                    cn_dict[i] = [None]
                    
        if model_type=="img2img":
            if consistory:
                raise "consistory don't support img2img now"
            d1, _, _, _ = image.size()
            if d1 == 1:
                image_load = [nomarl_upscale(image, width, height)]
            else:
                img_list = list(torch.chunk(image, chunks=d1))
                image_load = [nomarl_upscale(img, width, height) for img in img_list]

            gen = process_generation(pipe, image_load, model_type, steps, img_style, denoise_or_ip_sacle,
                                     style_strength_ratio, cfg,
                                     seed, id_length,
                                     character_prompt,
                                     negative_prompt,
                                     scene_prompts,
                                     width,
                                     height,
                                     load_chars,
                                     lora,
                                     trigger_words,photomake_mode,use_kolor,use_flux,make_dual_only,
                                     kolor_face,pulid,story_maker,input_id_emb_s_dict, input_id_img_s_dict,input_id_emb_un_dict, input_id_cloth_dict,guidance,condition_image,empty_emb_zero,use_cf,cf_scheduler,controlnet_path,controlnet_scale,cn_dict,input_tag_dict,SD35_mode,use_wrapper)

        else:
            if story_maker:
                raise "story maker only support img2img now"
            upload_images = None
            if consistory:
                if id_length>1:
                    raise "consistory support 1 role now "
                from .consistory.consistory_run import run_batch_generation,run_anchor_generation, run_extra_generation
                mask_dropout = 0.5
                same_latent = False
                n_achors = 2
                img_mode=False
                
                prompts_origin = scene_prompts.splitlines()
                prompts_origin = [i.strip() for i in prompts_origin]
                prompts_origin = [i for i in prompts_origin if '[' in i]  # 删除空行
                # print(prompts_origin)
                prompts = [prompt for prompt in prompts_origin if
                           not len(extract_content_from_brackets(prompt)) >= 2]  # 剔除双角色
                
                add_trigger_words = " " + trigger_words + " style "
                if lora:
                    if lora in lora_lightning_list:
                        prompts = remove_punctuation_from_strings(prompts)
                    else:
                        prompts = remove_punctuation_from_strings(prompts)
                        prompts = [item + add_trigger_words for item in prompts]
                character_dict_, character_list_ = character_to_dict(character_prompt, lora, add_trigger_words)
                
                clipped_prompts = prompts[:]
                nc_indexs = []
                for ind, prompt in enumerate(clipped_prompts):
                    if "[NC]" in prompt:
                        nc_indexs.append(ind)
                        if ind < id_length:
                            raise f"The first {id_length} row is id prompts, cannot use [NC]!"
                
                prompts = [
                    prompt if "[NC]" not in prompt else prompt.replace("[NC]", "")
                    for prompt in clipped_prompts
                ]
                
                if lora:
                    if lora in lora_lightning_list:
                        prompts = [
                            prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
                        ]
                    else:
                        prompts = [
                            prompt.rpartition("#")[0] + add_trigger_words if "#" in prompt else prompt for prompt in
                            prompts
                        ]
                else:
                    prompts = [
                        prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
                    ]
                
                (
                    character_index_dict_,
                    invert_character_index_dict_,
                    replace_prompts,
                    ref_indexs_dict_,
                    ref_totals_,
                ) = process_original_prompt(character_dict_, prompts, id_length, img_mode)
                
                if input_tag_dict:
                    if len(input_tag_dict) < len(replace_prompts):
                        raise "The number of input condition images is less than the number of scene prompts！"
                    replace_prompts = [prompt + " " + input_tag_dict[i] for i, prompt in enumerate(replace_prompts)]
                
                main_role = char_origin[0].replace("]", "").replace("[", "")
                concept_token = [main_role]
                if ")" in character_prompt:
                   object_role = character_prompt.split(")")[0].split("(")[-1]
                   concept_token=[main_role,object_role]
                style = "A photo of "
                subject=f"a {main_role} "
                replace_prompts= [f'{style}{subject} {i}' for i in replace_prompts]
                gpu = 0
                torch.cuda.reset_max_memory_allocated(gpu)
                if not cached:
                    if not inject:
                        pipe.enable_vae_slicing()
                        pipe.enable_model_cpu_offload()
                    else:
                        pipe.to(torch.float16)

                    anchor_out_images = run_batch_generation(pipe, replace_prompts, concept_token,negative_prompt, seed,n_steps=steps,
                                                         mask_dropout=mask_dropout, same_latent=same_latent, perform_injection=inject,n_achors=n_achors)
                else:
                    if len(replace_prompts)>2:
                        spilit_prompt=replace_prompts[:2]
                    else:
                        spilit_prompt=replace_prompts
                        
                    anchor_out_images, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(
                        pipe, spilit_prompt, concept_token,negative_prompt,
                        seed=seed, n_steps=steps, mask_dropout=mask_dropout, same_latent=same_latent,perform_injection=inject,
                        cache_cpu_offloading=True)
                    if len(replace_prompts) > 2:
                        left_prompt=replace_prompts[2:]
                    else:
                        left_prompt=replace_prompts[:1]  # use default
                    
                    for extra_prompt in left_prompt:
                        extra_image = run_extra_generation(pipe, [extra_prompt],
                                                                                 concept_token,negative_prompt,
                                                                                 anchor_cache_first_stage,
                                                                                 anchor_cache_second_stage,
                                                                                 seed=seed, n_steps=steps,
                                                                                 mask_dropout=mask_dropout,
                                                                                 same_latent=same_latent,
                                                                                 perform_injection=inject,
                                                                                 cache_cpu_offloading=True)
                        anchor_out_images.append(extra_image[0])
                #Report maximum GPU memory usage in GB
                max_memory_used = torch.cuda.max_memory_allocated(gpu) / (1024 ** 3)  # Convert to GB
                print(f"Maximum GPU memory used: {max_memory_used:.2f} GB")
    
                img=load_images_list(anchor_out_images)
                return (img, scene_prompts,)
            else:
                gen = process_generation(pipe, upload_images, model_type, steps, img_style, denoise_or_ip_sacle,
                                         style_strength_ratio, cfg,
                                         seed, id_length,
                                         character_prompt,
                                         negative_prompt,
                                         scene_prompts,
                                         width,
                                         height,
                                         load_chars,
                                         lora,
                                         trigger_words, photomake_mode, use_kolor, use_flux, make_dual_only, kolor_face,
                                         pulid, story_maker, input_id_emb_s_dict, input_id_img_s_dict,
                                         input_id_emb_un_dict, input_id_cloth_dict, guidance, condition_image,
                                         empty_emb_zero, use_cf, cf_scheduler, controlnet_path, controlnet_scale,
                                         cn_dict, input_tag_dict, SD35_mode, use_wrapper)
        
        for value in gen:
            print(type(value))
        image_pil_list = phi_list(value)

        image_pil_list_ms = image_pil_list.copy()
        if save_character:
            print("saving character...")
            save_results(pipe.unet)
        if prompts_dual:
            if not clip_vision_path:
                raise "need a clip_vison weight."
            if use_flux or use_kolor:
                raise "flux or kolor don't support MS diffsion."
            if model_type == "img2img":
                image_a = image_load[0]
                image_b = image_load[1]
            else:
                image_a = image_pil_list_ms[positions_char_1]
                image_b = image_pil_list_ms[positions_char_2]
            if story_maker:
                print("start sampler dual prompt using story maker")
                if make_dual_only:
                    del pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                    controlnet_path=None
                    pipe=story_maker_loader(clip_load,clip_vision_path,dir_path,ckpt_path,face_adapter,UniPCMultistepScheduler,controlnet_path,lora_scale,low_vram)
                mask_image_1=input_id_img_s_dict[role_name_list[0]][0]
                mask_image_2=input_id_img_s_dict[role_name_list[1]][0]
                
                if photomake_mode=="v2":
                    face_info_1 = input_id_emb_un_dict[role_name_list[0]][0]
                    face_info_2 = input_id_emb_un_dict[role_name_list[1]][0]
                else:
                    face_info_1 = input_id_emb_s_dict[role_name_list[0]][0]
                    face_info_2 = input_id_emb_s_dict[role_name_list[1]][0]
                
                cloth_info_1=None
                cloth_info_2=None
                if isinstance(condition_image,torch.Tensor):
                    cloth_info_1 = input_id_cloth_dict[role_name_list[0]][0]
                    cloth_info_2 = input_id_cloth_dict[role_name_list[1]][0]
                
                prompts_dual, negative_prompt = apply_style(
                    img_style, prompts_dual, negative_prompt
                )
                # 添加Lora trigger
                add_trigger_words = " " + trigger_words+ " style "
                if lora:
                    prompts_dual = remove_punctuation_from_strings(prompts_dual)
                    if lora not in lora_lightning_list:  # 加速lora不需要trigger
                        prompts_dual = [item + add_trigger_words for item in prompts_dual]
                
                prompts_dual = [item.replace(char_origin[0], char_describe[0]) for item in prompts_dual if
                                char_origin[0] in item]
                prompts_dual = [item.replace(char_origin[1], char_describe[1]) for item in prompts_dual if
                                char_origin[1] in item]
                
                prompts_dual = [item.replace("[", " ", ).replace("]", " ", ) for item in prompts_dual]
                image_dual = []
                if model_type=="txt2img":
                   setup_seed(seed)
                generator = torch.Generator(device=device).manual_seed(seed)
                for i,prompt in enumerate(prompts_dual):
                    output = pipe(
                        image=image_a, mask_image=mask_image_1, face_info=face_info_1,  # first person
                        image_2=image_b, mask_image_2=mask_image_2, face_info_2=face_info_2,  # second person
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        ip_adapter_scale=denoise_or_ip_sacle, lora_scale=lora_scale,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        height=height, width=width,
                        controlnet_conditioning_scale=controlnet_scale,
                        generator=generator,
                        cloth=cloth_info_1,
                        cloth_2=cloth_info_2
                    ).images[0]
                    image_dual.append(output)
            else: #using ms diffusion
                print("start sampler dual prompt using ms-diffusion")
                if controlnet_path:
                    if not isinstance(control_image,torch.Tensor):
                        raise "using controlnet need controlnet image input"
                if width != height:
                    square = max(height, width)
                    new_height, new_width = square, square
                    image_a = center_crop(image_a)
                    image_b = center_crop(image_b)
                else:
                    new_width = width
                    new_height = height
                #del pipe
                #gc.collect()
                #torch.cuda.empty_cache()
                from .model_loader_utils import msdiffusion_main
                image_dual = msdiffusion_main(image_a, image_b, prompts_dual, new_width, new_height, steps, seed,
                                              img_style, char_describe, char_origin, negative_prompt, clip_vision_path,
                                              model_type, lora, lora_path, lora_scale,
                                              trigger_words, ckpt_path, repo_id, guidance,
                                              mask_threshold, start_step, controlnet_path, control_image,
                                              controlnet_scale, cfg, guidance_list, scheduler_choice,pipe)
            j = 0
            for i in positions_dual:  # 重新将双人场景插入原序列
                if width != height:
                    img = center_crop_s(image_dual[j], width, height)
                else:
                    img = image_dual[j]
                image_pil_list.insert(int(i), img)
                j += 1
            image_list = narry_list(image_pil_list)
            torch.cuda.empty_cache()
        else:
            image_list = narry_list(image_pil_list)
        image = torch.from_numpy(np.fromiter(image_list, np.dtype((np.float32, (height, width, 3)))))
        if use_storydif and not prompts_dual:
            try:
               pipe.to("cpu")
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        return (image, scene_prompts,)
    
class Comic_Type:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "scene_prompts": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                             "fonts_list": (fonts_lists,),
                             "text_size": ("INT", {"default": 40, "min": 1, "max": 100}),
                             "comic_type": (["Four_Pannel", "Classic_Comic_Style"],),
                             "split_lines": ("STRING", {"default": "；"}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    ETURN_NAMES = ("image",)
    FUNCTION = "comic_gen"
    CATEGORY = "Storydiffusion"

    def comic_gen(self, image, scene_prompts, fonts_list, text_size, comic_type, split_lines):
        result = [item for index, item in enumerate(image)]
        total_results = narry_list_pil(result)
        font_choice = os.path.join(dir_path, "fonts", fonts_list)
        captions = scene_prompts.splitlines()
        if len(captions) > 1:
            captions = [caption.replace("(", "").replace(")", "") if "(" or ")" in caption else caption
                        for caption in captions]  # del ()
            captions = [caption.replace("[NC]", "") for caption in captions]
            captions = [caption.replace("]", "").replace("[", "") for caption in captions]
            captions = [
                caption.split("#")[-1] if "#" in caption else caption
                for caption in captions
            ]
        else:
            prompt_array = scene_prompts.replace(split_lines, "\n")
            captions = prompt_array.splitlines()
        font = ImageFont.truetype(font_choice, text_size)
        images = (
                get_comic(total_results, comic_type, captions=captions, font=font)
                + total_results
        )
        images = phi2narry(images[0])
        return (images,)


class Pre_Translate_prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scene_prompts": ("STRING", {"forceInput": True, "default": ""}),
                             "keep_character_name": ("BOOLEAN", {"default": False},)
                             }}

    RETURN_TYPES = ("STRING",)
    ETURN_NAMES = ("prompt_array",)
    FUNCTION = "translate_prompt"
    CATEGORY = "Storydiffusion"

    def translate_prompt(self, scene_prompts, keep_character_name):
        captions = scene_prompts.splitlines()
        if not keep_character_name:
            captions = [caption.split(")", 1)[-1] if ")" in caption else caption
                        for caption in captions]  # del character
        else:
            captions = [caption.replace("(", "").replace(")", "") if "(" or ")" in caption else caption
                        for caption in captions]  # del ()
        captions = [caption.replace("[NC]", "") for caption in captions]
        if not keep_character_name:
            captions = [caption.split("]", 1)[-1] for caption in captions]
        else:
            captions = [caption.replace("]", "").replace("[", "") for caption in captions]
        captions = [
            caption.split("#")[-1] if "#" in caption else caption
            for caption in captions
        ]
        scene_prompts = ''.join(captions)
        return (scene_prompts,)

class EasyFunction_Lite:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo": ("STRING", { "default": "F:/test/ComfyUI/models/diffusers/pzc163/MiniCPMv2_6-prompt-generator"}),
                             "function_mode": (["tag", "clip","mask","llm",],),
                             "select_method":("STRING",{ "default":""}),
                             "temperature": (
                                 "FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                             }}
    
    RETURN_TYPES = ("MODEL",)
    ETURN_NAMES = ("model",)
    FUNCTION = "easy_function_main"
    CATEGORY = "Storydiffusion"
    
    def easy_function_main(self, repo,function_mode,select_method,temperature):
        if function_mode=="tag":
            from .model_loader_utils import StoryLiteTag
            model = StoryLiteTag(device, temperature,select_method, repo)
        elif function_mode=="clip":
            model=None
        elif function_mode=="mask":
            model=None
        else:
            model=None
        return (model,)


NODE_CLASS_MAPPINGS = {
    "Storydiffusion_Model_Loader": Storydiffusion_Model_Loader,
    "Storydiffusion_Sampler": Storydiffusion_Sampler,
    "Pre_Translate_prompt": Pre_Translate_prompt,
    "Comic_Type": Comic_Type,
    "EasyFunction_Lite":EasyFunction_Lite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Storydiffusion_Model_Loader": "Storydiffusion_Model_Loader",
    "Storydiffusion_Sampler": "Storydiffusion_Sampler",
    "Pre_Translate_prompt": "Pre_Translate_prompt",
    "Comic_Type": "Comic_Type",
    "EasyFunction_Lite":"EasyFunction_Lite"
}
