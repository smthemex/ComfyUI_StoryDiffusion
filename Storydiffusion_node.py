# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import hashlib
import numpy as np
import torch
import gc
import copy
import os
import random

import yaml
from PIL import ImageFont
from ip_adapter.attention_processor import IPAttnProcessor2_0
import sys
import re
from .utils.gradio_utils import (
    character_to_dict,
    process_original_prompt,
    get_ref_character,
    cal_attn_mask_xl,
    cal_attn_indice_xl_effcient_memory,
    is_torch2_available,
)
from PIL import Image

if is_torch2_available():
    from .utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from .utils.gradio_utils import AttnProcessor
from huggingface_hub import hf_hub_download
from diffusers import (StableDiffusionXLPipeline, DiffusionPipeline, DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler,
                       AutoPipelineForInpainting, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, UNet2DConditionModel,
                       AutoPipelineForText2Image, StableDiffusionXLControlNetImg2ImgPipeline, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, DDPMScheduler, LCMScheduler)
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

from .msdiffusion.models.projection import Resampler
from .msdiffusion.models.model import MSAdapter
from .msdiffusion.utils import get_phrase_idx, get_eot_idx
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch.nn.functional as F
from .utils.utils import get_comic
from .utils.style_template import styles
from .utils.load_models_utils import get_models_dict, load_models, get_instance_path, get_lora_dict
import folder_paths
from comfy.utils import common_upscale

global total_count, attn_count, cur_step, mask1024, mask4096, attn_procs, unet
global sa32, sa64
global write
global height, width
STYLE_NAMES = list(styles.keys())

global models_dict
models_dict = get_models_dict()  # 获取模型信息
import diffusers

dif_version = str(diffusers.__version__)
dif_version_int = int(dif_version.split(".")[1])
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

fonts_path = os.path.join(dir_path, "fonts")
fonts_lists = os.listdir(fonts_path)

yaml_list = list(models_dict.keys())

global lora_get
lora_get = get_lora_dict()
lora_lightning_list = lora_get["lightning_xl_lora"]
# print(lora_lightning_list)

scheduler_list = [
    "Euler", "Euler a", "DDIM", "DDPM", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras",
    "DPM++ SDE", "DPM++ SDE Karras", "DPM2",
    "DPM2 Karras", "DPM2 a", "DPM2 a Karras", "Heun", "LCM", "LMS", "LMS Karras", "UniPC"
]


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def add_pil(list, list_add, num):
    new_list = list[:num] + list_add + list[num:]
    return new_list


def tensor_to_image(tensor):
    # tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


# get fonts list
def has_parentheses(s):
    return bool(re.search(r'\(.*?\)', s))


def contains_brackets(s):
    return '[' in s or ']' in s


def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = phi2narry(value)
        list_in[i] = modified_value
    return list_in


def remove_punctuation_from_strings(lst):
    pattern = r"[\W]+$"  # 匹配字符串末尾的所有非单词字符
    return [re.sub(pattern, '', s) for s in lst]


def format_punctuation_from_strings(lst):
    pattern = r"[\W]+$"  # 匹配字符串末尾的所有非单词字符
    return [re.sub(pattern, ';', s) for s in lst]


def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in


def narry_list_pil(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = tensor_to_image(value)
        list_in[i] = modified_value
    return list_in


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path


def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list


def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[style_name])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[style_name])
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative


def load_character_files(character_files: str):
    if character_files == "":
        raise "Please set a character file!"
    character_files_arr = character_files.splitlines()
    primarytext = []
    for character_file_name in character_files_arr:
        character_file = torch.load(
            character_file_name, map_location=torch.device("cpu")
        )
        primarytext.append(character_file["character"] + character_file["description"])
    return array2string(primarytext)


def array2string(arr):
    stringtmp = ""
    for i, part in enumerate(arr):
        if i != len(arr) - 1:
            stringtmp += part + "\n"
        else:
            stringtmp += part

    return stringtmp


def load_character_files_on_running(unet, character_files: str):
    if character_files == "":
        return False
    character_files_arr = character_files.splitlines()
    for character_file in character_files_arr:
        load_single_character_weights(unet, character_file)
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
            weights_to_save[attn_name] = {}
            for step_key in attn_processor.id_bank[character].keys():
                weights_to_save[attn_name][step_key] = [
                    tensor.cpu()
                    for tensor in attn_processor.id_bank[character][step_key]
                ]
    # 使用torch.save保存权重
    torch.save(weights_to_save, filepath)


def load_single_character_weights(unet, filepath):
    """
    从指定文件中加载权重到 attention_processor 类的 id_bank 中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重文件的路径。
    """
    # 使用torch.load来读取权重
    weights_to_load = torch.load(filepath, map_location=torch.device("cpu"))
    character = weights_to_load["character"]
    description = weights_to_load["description"]
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 转移权重到GPU（如果GPU可用的话）并赋值给id_bank
            attn_processor.id_bank[character] = {}
            for step_key in weights_to_load[attn_name].keys():
                attn_processor.id_bank[character][step_key] = [
                    tensor.to(unet.device)
                    for tensor in weights_to_load[attn_name][step_key]
                ]


def get_scheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler


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


def save_results(unet, img_list):
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # folder_name = f"{timestamp}"
    weight_folder_name = os.path.join(dir_path, "weights")
    # weight_folder_name = f"{folder_name}/weights"
    # os.makedirs(weight_folder_name)
    # 创建文件夹
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    #     os.makedirs(weight_folder_name)
    file_prefix = ''.join(random.choice("0123456789") for _ in range(5))
    for idx, img in enumerate(img_list):
        # file_path_res = os.path.join(file_path, "output", f"image_{file_prefix}_{idx}.png")  # 图片文件名
        # img.save(file_path_res)
        img_enumerate = img
        img_idx = idx
    global character_dict

    # return img
    # for char in character_dict:
    #     description = character_dict[char]
    #     save_single_character_weights(unet,char,description,os.path.join(weight_folder_name, f'{char}.pt'))


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
        global height, width
        global character_dict, character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals, cur_character
        if attn_count == 0 and cur_step == 0:
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )
        if write:
            assert len(cur_character) == 1
            if hidden_states.shape[1] == (height // 32) * (width // 32):
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
                if hidden_states.shape[1] == (height // 32) * (width // 32):
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
                height,
                width,
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
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
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
                batch_size, channel, height, width
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
        _Ip_Adapter_Strength,
        _style_strength_ratio,
        guidance_scale,
        seed_,
        id_length,
        general_prompt,
        negative_prompt,
        prompt_array,
        width,
        height,
        _char_files,
        lora,
        trigger_words
):  # Corrected font_choice usage
    if len(general_prompt.splitlines()) >= 3:
        raise "Support for more than three characters is temporarily unavailable due to VRAM limitations, but this issue will be resolved soon."
    # _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    if model_type == "Photomaker" and "img" not in general_prompt:
        raise 'Please add the triger word " img "  behind the class word you want to customize, such as: man img or woman img'

    global total_length, attn_procs, unet, cur_model_type
    global write
    global cur_step, attn_count

    load_chars = load_character_files_on_running(unet, character_files=_char_files)

    prompts_origin = prompt_array.splitlines()
    prompts = [prompt for prompt in prompts_origin if not has_parentheses(prompt)]  # 剔除双角色
    add_trigger_words = "," + trigger_words + " " + "style" + ";"
    if lora != "none":
        if lora in lora_lightning_list:
            prompts = remove_punctuation_from_strings(prompts)
        else:
            prompts = remove_punctuation_from_strings(prompts)
            prompts = [item + add_trigger_words for item in prompts]

    global character_dict, character_index_dict, invert_character_index_dict, ref_indexs_dict, ref_totals

    character_dict, character_list = character_to_dict(general_prompt, lora, add_trigger_words)
    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    generator = torch.Generator(device=device).manual_seed(seed_)
    # sa32, sa64 = sa32_, sa64_
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
    if lora != "none":
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

    # id_prompts = prompts[:id_length]
    (
        character_index_dict,
        invert_character_index_dict,
        replace_prompts,
        ref_indexs_dict,
        ref_totals,
    ) = process_original_prompt(character_dict, prompts.copy(), id_length)
    if model_type != "original":
        # _upload_images = [_upload_images]
        input_id_images_dict = {}
        if len(upload_images) != len(character_dict.keys()):
            raise f"You upload images({len(upload_images)}) is not equal to the number of characters({len(character_dict.keys())})!"
        for ind, img in enumerate(upload_images):
            input_id_images_dict[character_list[ind]] = [img]  # 已经pil转化了 不用load
            # input_id_images_dict[character_list[ind]] = [load_image(img)]
    # real_prompts = prompts[id_length:]
    if device == "cuda":
        torch.cuda.empty_cache()
    write = True
    cur_step = 0
    attn_count = 0
    # id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    setup_seed(seed_)
    total_results = []
    id_images = []
    results_dict = {}
    global cur_character
    if not load_chars:
        for character_key in character_dict.keys():
            cur_character = [character_key]
            ref_indexs = ref_indexs_dict[character_key]
            current_prompts = [replace_prompts[ref_ind] for ref_ind in ref_indexs]
            setup_seed(seed_)
            generator = torch.Generator(device=device).manual_seed(seed_)
            cur_step = 0
            cur_positive_prompts, negative_prompt = apply_style(
                style_name, current_prompts, negative_prompt
            )
            if model_type == "original":
                id_images = pipe(
                    cur_positive_prompts,
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images

            elif model_type == "Photomaker":
                id_images = pipe(
                    cur_positive_prompts,
                    input_id_images=input_id_images_dict[character_key],
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    start_merge_step=start_merge_step,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            else:
                raise NotImplementedError(
                    "You should choice between original and Photomaker!",
                    f"But you choice {model_type}",
                )

            # total_results = id_images + total_results
            # yield total_results
            for ind, img in enumerate(id_images):
                results_dict[ref_indexs[ind]] = img
            # real_images = []
            yield [results_dict[ind] for ind in results_dict.keys()]
    write = False
    if not load_chars:
        real_prompts_inds = [
            ind for ind in range(len(prompts)) if ind not in ref_totals
        ]
    else:
        real_prompts_inds = [ind for ind in range(len(prompts))]
    for real_prompts_ind in real_prompts_inds:  # 非角色流程
        real_prompt = replace_prompts[real_prompts_ind]
        cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        setup_seed(seed_)
        if len(cur_character) > 1 and model_type == "Photomaker":
            raise "Temporarily Not Support Multiple character in Ref Image Mode!"
        generator = torch.Generator(device=device).manual_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        if model_type == "original":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        elif model_type == "Photomaker":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                input_id_images=(
                    input_id_images_dict[cur_character[0]]
                    if real_prompts_ind not in nc_indexs
                    else input_id_images_dict[character_list[0]]
                ),
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                start_merge_step=start_merge_step,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
                nc_flag=True if real_prompts_ind in nc_indexs else False,
            ).images[0]
        else:
            raise NotImplementedError(
                "You should choice between original and Photomaker!",
                f"But you choice {model_type}",
            )
        yield [results_dict[ind] for ind in results_dict.keys()]
    total_results = [results_dict[ind] for ind in range(len(prompts))]
    yield total_results


def msdiffusion_main(pipe, image_1, image_2, prompts_dual, width, height, steps, seed, style_name, negative_prompt,
                     encoder_repo, _model_type, _sd_type, lora, lora_path, lora_scale, trigger_words, ckpt_path,
                     original_config_file, role_scale, mask_threshold, start_step):
    def get_phrases_idx(tokenizer, phrases, prompt):
        res = []
        phrase_cnt = {}
        for phrase in phrases:
            if phrase in phrase_cnt:
                cur_cnt = phrase_cnt[phrase]
                phrase_cnt[phrase] += 1
            else:
                cur_cnt = 0
                phrase_cnt[phrase] = 1
            res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
        return res

    if _model_type == "Using Ref Images":  # 图生图双角色目前只能先用原方法，2者encoder模型不同，没法相互调用
        del pipe
        # load SDXL pipeline
        if _sd_type == "Use_Single_XL_Model":
            sd_model_path = ckpt_path
        else:
            sd_model_path = models_dict[_sd_type]["path"]  # diffuser models
        single_files = models_dict[_sd_type]["single_files"]

        if single_files:
            if dif_version_int >= 28:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    sd_model_path, original_config=original_config_file, torch_dtype=torch.float16,
                    add_watermarker=False, )
            else:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    sd_model_path, original_config_file=original_config_file, torch_dtype=torch.float16,
                    add_watermarker=False,
                )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                sd_model_path, torch_dtype=torch.float16, add_watermarker=False,
            )

        pipe.to("cuda")
        if lora != "none":
            if lora in lora_lightning_list:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
            else:
                pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                pipe.fuse_lora(adapter_names=[trigger_words, ], lora_scale=lora_scale)

    device = "cuda"
    ms_ckpt = get_instance_path(os.path.join(dir_path, "weights", "ms_adapter.bin"))
    image_processor = CLIPImageProcessor()

    if encoder_repo.count("/") > 1:
        encoder_repo = get_instance_path(encoder_repo)

    image_encoder_type = "clip"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(encoder_repo).to(device, dtype=torch.float16)
    image_encoder_projection_dim = image_encoder.config.projection_dim
    num_tokens = 16
    image_proj_type = "resampler"
    latent_init_mode = "grounding"
    # latent_init_mode = "random"
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=pipe.unet.config.cross_attention_dim,
        ff_mult=4,
        latent_init_mode=latent_init_mode,
        phrase_embeddings_dim=pipe.text_encoder.config.projection_dim,
    ).to(device, dtype=torch.float16)
    ms_model = MSAdapter(pipe.unet, image_proj_model, ckpt_path=ms_ckpt, device=device, num_tokens=num_tokens)
    ms_model.to(device, dtype=torch.float16)

    input_images = [image_1, image_2]
    input_images = [x.convert("RGB").resize((width, height)) for x in input_images]

    # generation configs
    num_samples = 1
    image_ouput = []
    # get role name
    role_name = prompts_dual[0]
    role_name = role_name.split(")")[0]
    role_name = role_name.split("(")[-1]
    role_name = role_name.split("and")
    role_a = role_name[0]
    role_b = role_name[1]
    # get n p prompt
    prompts_dual, negative_prompt = apply_style(
        style_name, prompts_dual, negative_prompt
    )
    # 添加Lora trigger
    add_trigger_words = "," + trigger_words + " " + "style" + ";"
    if lora != "none":
        prompts_dual = remove_punctuation_from_strings(prompts_dual)
        if lora not in lora_lightning_list:  # 加速lora不需要trigger
            prompts_dual = [item + add_trigger_words for item in prompts_dual]

    for i, prompt in enumerate(prompts_dual):
        # print(i, prompt)
        prompt = prompt.replace("(", "").replace(")", "")
        # boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # man+women
        boxes = [[[0., 0., 0., 0.], [0., 0., 0., 0.]]]  # used if you want no layout guidance
        phrases = [[role_a, role_b]]
        drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens

        # used to get the attention map, return zero if the phrase is not in the prompt
        phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
        eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
        # print(phrase_idxes, eot_idxes)

        images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples,
                                   num_inference_steps=steps,
                                   seed=seed,
                                   prompt=[prompt], negative_prompt=negative_prompt, scale=role_scale,
                                   image_encoder=image_encoder,
                                   image_processor=image_processor, boxes=boxes, mask_threshold=mask_threshold,
                                   start_step=start_step,
                                   image_proj_type=image_proj_type, image_encoder_type=image_encoder_type,
                                   phrases=phrases,
                                   drop_grounding_tokens=drop_grounding_tokens,
                                   phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=height, width=width)

        image_ouput += images
    return image_ouput


class Storydiffusion_Model_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_type": (yaml_list,),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "photomake_model": (["none"] + folder_paths.get_filename_list("photomaker"),),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_words": ("STRING", {"default": "best quality"}),
                "scheduler": (scheduler_list,),
                "model_type": (["Photomaker", "original"],),
                "id_number": ("INT", {"default": 2, "min": 1, "max": 2, "step": 1, "display": "number"}),
                "sa32_degree": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "sa64_degree": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "img_width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32, "display": "number"}),
                "img_height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32, "display": "number"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("pipe", "info",)
    FUNCTION = "story_model_loader"
    CATEGORY = "Storydiffusion"

    def story_model_loader(self, sd_type, ckpt_name, photomake_model, lora, lora_scale, scheduler, trigger_words,
                           model_type, sa32_degree, sa64_degree, id_number, img_width, img_height):

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        ckpt_path = get_instance_path(ckpt_path)
        scheduler_choice = get_scheduler(scheduler)

        local_dir = os.path.join(file_path, "models", "photomaker")
        photomaker_local_path = get_instance_path(os.path.join(local_dir, photomake_model))
        if photomake_model == "none":
            photomaker_path = hf_hub_download(
                repo_id="TencentARC/PhotoMaker",
                filename="photomaker-v1.bin",
                repo_type="model",
                local_dir=local_dir,
            )
        else:
            photomaker_path = photomaker_local_path

        if lora != "none":
            lora_path = folder_paths.get_full_path("loras", lora)
            lora_path = get_instance_path(lora_path)
        else:
            lora_path = ""
        if "/" in lora:
            lora = lora.split("/")[-1]
        if "\\" in lora:
            lora = lora.split("\\")[-1]

        # global
        global mask1024, mask4096, attn_procs, unet, pipe
        global sa32, sa64, write, height, width, sd_model_path
        global attn_count, total_count, id_length, total_length, cur_step, cur_model_type

        sa32 = sa32_degree
        sa64 = sa64_degree
        attn_count = 0
        total_count = 0
        cur_step = 0
        id_length = id_number
        total_length = 5
        cur_model_type = "Unstable" + "-" + "original"
        attn_procs = {}
        write = False
        height = img_height
        width = img_width

        # load model
        original_config_file = os.path.join(dir_path, 'config', 'sd_xl_base.yaml')
        original_config_file = get_instance_path(original_config_file)
        if sd_type == "Use_Single_XL_Model":
            sd_model_path = ckpt_path
        else:
            sd_model_path = models_dict[sd_type]["path"]  # diffuser models
        single_files = models_dict[sd_type]["single_files"]
        ### LOAD Stable Diffusion Pipeline
        if single_files:
            if dif_version_int >= 28:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    sd_model_path, original_config=original_config_file, torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    sd_model_path, original_config_file=original_config_file, torch_dtype=torch.float16
                )
        else:
            if sd_type == "Playground_v2p5":
                pipe = DiffusionPipeline.from_pretrained(
                    sd_model_path, torch_dtype=torch.float16
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    sd_model_path, torch_dtype=torch.float16
                )
        pipe = pipe.to(device)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(50)
        pipe.enable_vae_slicing()
        if device != "mps":
            pipe.enable_model_cpu_offload()
        unet = pipe.unet

        ### Insert PairedAttention
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None and (name.startswith("up_blocks")):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
                total_count += 1
            else:
                attn_procs[name] = AttnProcessor()
        print("successsfully load paired self-attention")
        print(f"number of the processor : {total_count}")
        unet.set_attn_processor(copy.deepcopy(attn_procs))
        mask1024, mask4096 = cal_attn_mask_xl(
            total_length,
            id_length,
            sa32,
            sa64,
            height,
            width,
            device=device,
            dtype=torch.float16,
        )

        # pipe
        use_safe_tensor = True
        for attn_processor in pipe.unet.attn_processors.values():
            if isinstance(attn_processor, SpatialAttnProcessor2_0):
                for values in attn_processor.id_bank.values():
                    del values
                attn_processor.id_bank = {}
                attn_processor.id_length = id_length
                attn_processor.total_length = id_length + 1
        gc.collect()
        torch.cuda.empty_cache()

        if sd_type == "Playground_v2p5":
            cur_model_type = "Playground_v2p5" + "-" + model_type
        if sd_type == "Unstable":
            cur_model_type = "Unstable" + "-" + model_type

        if cur_model_type != sd_type + "-" + model_type:  # load stable model
            # apply the style template
            ##### load pipe
            del pipe
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            model_info = models_dict[sd_type]
            model_info["model_type"] = model_type
            if sd_type == "Use_Single_XL_Model":
                model_info["path"] = ckpt_path
            pipe = load_models(model_info, sd_type, device=device, photomaker_path=photomaker_path, lora=lora,
                               lora_path=lora_path,
                               trigger_words=trigger_words, lora_scale=lora_scale)
            set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            ##### ########################
            pipe.scheduler = scheduler_choice.from_config(pipe.scheduler.config)
            # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            cur_model_type = sd_type + "-" + model_type
            pipe.enable_vae_slicing()
            if device != "mps":
                pipe.enable_model_cpu_offload()
        else:  # add  lora
            if lora != "none":
                if lora in lora_lightning_list:
                    pipe.load_lora_weights(lora_path)
                    pipe.fuse_lora()
                else:
                    pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                    pipe.fuse_lora(adapter_names=[trigger_words, ], lora_scale=lora_scale)
            unet = pipe.unet
            # unet.set_attn_processor(copy.deepcopy(attn_procs))
        info = str(";".join(
            [model_type, ckpt_path, lora_path, original_config_file, lora, trigger_words, sd_type, str(lora_scale)]))
        return (pipe, info,)


class Storydiffusion_Text2Img:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "character_prompt": ("STRING", {"multiline": True,
                                                "default": "[Bob] A man, wearing a black sui,\n"
                                                           "[Alice]a woman, wearing a white shirt."}),
                "scene_prompts": ("STRING", {"multiline": True,
                                             "default": "[Bob] at home, read new paper #at home, The newspaper says "
                                                        "there is a treasure house in the forest;\n[Bob] on the road, "
                                                        "near the forest;\n[Alice] is make a call at home;\n[NC]A tiger "
                                                        "appeared in the forest, at night;\n[NC] The car on the road, "
                                                        "near the forest #They drives to the forest in search of "
                                                        "treasure house;\n[Bob]at night #He is overjoyed inside the house;\n[Alice] very frightened, open mouth, in the forest, at night."}),
                "split_prompt":("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "bad anatomy, bad hands, missing fingers, extra fingers, "
                                                          "three hands, three legs, bad arms, missing legs, "
                                                          "missing arms, poorly drawn face, bad face, fused face, "
                                                          "cloned face, three crus, fused feet, fused thigh, "
                                                          "extra crus, ugly fingers, horn,"
                                                          "animate, amputation, disconnected limbs"}),
                "img_style": (
                    ["No_style", "Realistic", "Japanese_Anime", "Digital_Oil_Painting", "Pixar_Disney_Character",
                     "Photographic", "Comic_book",
                     "Line_art", "Black_and_White_Film_Noir", "Isometric_Rooms"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "encoder_repo": ("STRING", {"default": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"}),
                "role_scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "mask_threshold": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "start_step": ("INT", {"default": 5, "min": 1, "max": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "scene_prompts",)
    FUNCTION = "text2image"
    CATEGORY = "Storydiffusion"

    def text2image(self, pipe, info, character_prompt, negative_prompt, scene_prompts, split_prompt,img_style, seed, steps, cfg,
                   encoder_repo,
                   role_scale, mask_threshold, start_step):
        model_type, ckpt_path, lora_path, original_config_file, lora, trigger_words, sd_type, lora_scale = info.split(
            ";")
        lora_scale = float(lora_scale)

        #格式化文字内容
        if split_prompt:
            scene_prompts.replace("\n", "").replace(split_prompt, ";\n").strip()
            character_prompt.replace("\n", "").replace(split_prompt, ";\n").strip()
        else:
            scene_prompts.strip()
            character_prompt.strip()
            if "\n" not in scene_prompts:
                scene_prompts.replace(";", ";\n").strip()
            if "\n" in character_prompt:
                if character_prompt.count("\n")>1:
                    character_prompt.replace("\n", "").replace("[", "\n[").strip()
                    if character_prompt.count("\n") > 1:
                        character_prompt.replace("\n", "").replace("[", "\n[",2).strip()# 多行角色在这里强行转为双角色

        # 从角色列表获取角色方括号信息
        char_origin = character_prompt.splitlines()
        char_origin = [char.split("]")[0] + "]" for char in char_origin]

        prompts_origin = scene_prompts.splitlines()

        # 判断是否有双角色prompt，如果有，获取双角色列表及对应的位置列表，
        positions_dual = [index for index, prompt in enumerate(prompts_origin) if has_parentheses(prompt)]
        prompts_dual = [prompt for prompt in prompts_origin if has_parentheses(prompt)]

        if len(char_origin) == 2:
            positions_char_1 = [index for index, prompt in enumerate(prompts_origin) if char_origin[0] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置
            positions_char_2 = [index for index, prompt in enumerate(prompts_origin) if char_origin[1] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置

        upload_images = None
        _Ip_Adapter_Strength = 0.5
        _style_strength_ratio = 20
        char_files = ""

        gen = process_generation(pipe, upload_images, model_type, steps, img_style, _Ip_Adapter_Strength,
                                 _style_strength_ratio, cfg,
                                 seed, id_length,
                                 character_prompt,
                                 negative_prompt,
                                 scene_prompts,
                                 width,
                                 height,
                                 char_files,
                                 lora,
                                 trigger_words, )

        for value in gen:
            pass_value = value
            del pass_value
        image_pil_list = phi_list(value)

        if prompts_dual:
            image_1 = image_pil_list[positions_char_1]
            image_2 = image_pil_list[positions_char_2]
            # del pipe
            image_dual = msdiffusion_main(pipe, image_1, image_2, prompts_dual, width, height, steps, seed,
                                          img_style, negative_prompt, encoder_repo, model_type, sd_type, lora,
                                          lora_path,
                                          lora_scale, trigger_words, ckpt_path, original_config_file, role_scale,
                                          mask_threshold, start_step)
            j = 0
            for i in positions_dual:
                img = image_dual[j]
                image_pil_list.insert(int(i), img)
                j += 1
            image_list = narry_list(image_pil_list)
        else:
            image_list = narry_list(image_pil_list)
        image = torch.from_numpy(np.fromiter(image_list, np.dtype((np.float32, (height, width, 3)))))
        del pipe
        del gen
        return (image, scene_prompts,)


class Storydiffusion_Img2Img:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipe": ("MODEL",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "character_prompt": ("STRING", {"multiline": True,
                                                "default": "[Taylor]a woman img, wearing a white T-shirt, blue loose hair.\n"
                                                           "[Lecun] a man img,wearing a suit,black hair."}),
                "scene_prompts": ("STRING", {"multiline": True,
                                             "default": "[Taylor]wake up in the bed;\n[Taylor]have breakfast by the window;\n[Lecun]is walking on the road, go to company;\n[Lecun]work in the company."}),
                "split_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "bad anatomy, bad hands, missing fingers, extra fingers, "
                                                          "three hands, three legs, bad arms, missing legs, "
                                                          "missing arms, poorly drawn face, bad face, fused face, "
                                                          "cloned face, three crus, fused feet, fused thigh, "
                                                          "extra crus, ugly fingers, horn,"
                                                          "animate, amputation, disconnected limbs"}),
                "img_style": (
                    ["No_style", "Realistic", "Japanese_Anime", "Digital_Oil_Painting", "Pixar_Disney_Character",
                     "Photographic", "Comic_book",
                     "Line_art", "Black_and_White_Film_Noir", "Isometric_Rooms"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "ip_adapter_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 10, "max": 50, "step": 1, "display": "number"}),
                "encoder_repo": ("STRING", {"default": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"}),
                "role_scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "mask_threshold": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "start_step": ("INT", {"default": 5, "min": 1, "max": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "prompt_array",)
    FUNCTION = "img2image"
    CATEGORY = "Storydiffusion"

    def img2image(self, image, pipe, info, character_prompt, negative_prompt, scene_prompts, split_prompt,img_style, seed, steps,
                  cfg, ip_adapter_strength, style_strength_ratio, encoder_repo,
                  role_scale, mask_threshold, start_step):
        model_type, ckpt_path, lora_path, original_config_file, lora, trigger_words, sd_type, lora_scale = info.split(
            ";")
        lora_scale = float(lora_scale)
        d1, _, _, _ = image.size()
        if d1 == 2:
            image_1, image_2 = torch.chunk(image, chunks=2)
            image_1 = tensor_to_image(image_1)
            image_2 = tensor_to_image(image_2)
            image_load = [image_1] + [image_2]
        elif d1 == 3:
            image_1, image_2, image_3 = torch.chunk(image, chunks=3)
            image_1 = tensor_to_image(image_1)
            image_2 = tensor_to_image(image_2)
            image_3 = tensor_to_image(image_3)
            image_load = [image_1] + [image_2] + [image_3]
        else:
            image = tensor_to_image(image)
            image_load = [image]

        #格式化文字内容
        if split_prompt:
            scene_prompts.replace("\n", "").replace(split_prompt, ";\n").strip()
            character_prompt.replace("\n", "").replace(split_prompt, ";\n").strip()
        else:
            scene_prompts.strip()
            character_prompt.strip()
            if "\n" not in scene_prompts:
                scene_prompts.replace(";", ";\n").strip()
            if "\n" in character_prompt:
                if character_prompt.count("\n")>1:
                    character_prompt.replace("\n", "").replace("[", "\n[").strip()
                    if character_prompt.count("\n") > 1:
                        character_prompt.replace("\n", "").replace("[", "\n[",2).strip()# 多行角色在这里强行转为双角色

        # 从角色列表获取角色方括号信息
        char_origin = character_prompt.splitlines()
        char_origin = [char.split("]")[0] + "]" for char in char_origin]


        # 判断是否有双角色prompt，如果有，获取双角色列表及对应的位置列表，
        prompts_origin = scene_prompts.splitlines()
        positions_dual = [index for index, prompt in enumerate(prompts_origin) if has_parentheses(prompt)]
        prompts_dual = [prompt for prompt in prompts_origin if has_parentheses(prompt)]

        if len(char_origin) == 2:
            positions_char_1 = [index for index, prompt in enumerate(prompts_origin) if char_origin[0] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置
            positions_char_2 = [index for index, prompt in enumerate(prompts_origin) if char_origin[1] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置

        char_files = ""

        gen = process_generation(pipe, image_load, model_type, steps, img_style, ip_adapter_strength,
                                 style_strength_ratio, cfg,
                                 seed, id_length,
                                 character_prompt,
                                 negative_prompt,
                                 scene_prompts,
                                 width,
                                 height,
                                 char_files,
                                 lora,
                                 trigger_words, )

        for value in gen:
            pass_value = value
            del pass_value
        image_pil_list = phi_list(value)

        if prompts_dual:
            image_1 = image_pil_list[positions_char_1]
            image_2 = image_pil_list[positions_char_2]
            # del pipe
            image_dual = msdiffusion_main(pipe, image_1, image_2, prompts_dual, width, height, steps, seed,
                                          img_style, negative_prompt, encoder_repo, model_type, sd_type, lora,
                                          lora_path,
                                          lora_scale, trigger_words, ckpt_path, original_config_file, role_scale,
                                          mask_threshold, start_step)
            j = 0
            for i in positions_dual:
                img = image_dual[j]
                image_pil_list.insert(int(i), img)
                j += 1
            image_list = narry_list(image_pil_list)
        else:
            image_list = narry_list(image_pil_list)
        image = torch.from_numpy(np.fromiter(image_list, np.dtype((np.float32, (height, width, 3)))))
        del pipe
        del gen
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


NODE_CLASS_MAPPINGS = {
    "Storydiffusion_Model_Loader": Storydiffusion_Model_Loader,
    "Storydiffusion_Text2Img": Storydiffusion_Text2Img,
    "Storydiffusion_Img2Img": Storydiffusion_Img2Img,
    "Pre_Translate_prompt": Pre_Translate_prompt,
    "Comic_Type": Comic_Type
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Storydiffusion_Model_Loader": "Storydiffusion_Model_Loader",
    "Storydiffusion_Text2Img": "Storydiffusion_Text2Img",
    "Storydiffusion_Img2Img": "Storydiffusion_Img2Img",
    "Pre_Translate_prompt": "Pre_Translate_prompt",
    "Comic_Type": "Comic_Type"

}
