# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import logging
import numpy as np
import torch
import os
from PIL import ImageFont,Image
from .StoryMaker.pipeline_sdxl_storymaker_wrapper import encode_prompt_image_emb_
import folder_paths
from comfy.model_management import total_vram
import comfy
import latent_preview
import torchvision.transforms.functional as TVF
from .utils.utils import get_comic
from .model_loader_utils import  (phi2narry,Loader_storydiffusion,Loader_Flux_Pulid,load_pipeline_consistory,Loader_KOLOR,replicate_data_by_indices,glm_single_encode,
                                  get_float,gc_cleanup,tensor_to_image,photomaker_clip,Loader_UNO,tensortopil_list_upscale,tensortopil_list,extract_content_from_brackets_,
                                  narry_list_pil,pre_text2infer,cf_clip,get_phrases_idx_cf,get_eot_idx_cf,get_ms_phrase_emb,get_extra_function,photomaker_clip_v2,adjust_indices,
                                  get_scheduler,apply_style_positive,fitter_cf_model_type,Infer_MSdiffusion,Loader_story_maker,Loader_InfiniteYou,
                                  nomarl_upscale,SAMPLER_NAMES,SCHEDULER_NAMES,lora_lightning_list)
from .utils.gradio_utils import cal_attn_indice_xl_effcient_memory,is_torch2_available
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


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

infer_type_g=torch.float16 if device=="cuda" else torch.float32 #？

MAX_SEED = np.iinfo(np.int32).max

dir_path = os.path.dirname(os.path.abspath(__file__))
fonts_lists = os.listdir(os.path.join(dir_path, "fonts"))
photomaker_dir=os.path.join(folder_paths.models_dir, "photomaker")
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) # use gguf dir

    
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
        global  character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals
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
    RETURN_NAMES = ("image",)
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
            captions = [caption.split(")", 1)[-1] if ")" in caption else caption for caption in captions]  # del character
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
        return {"required": {"extra_repo": ("STRING", { "default": ""}),
                            "checkpoints": (["none"] + folder_paths.get_filename_list("diffusion_models")+folder_paths.get_filename_list("gguf")+folder_paths.get_filename_list("clip"),),
                            "clip_vision": (["none"] + folder_paths.get_filename_list("clip_vision")+folder_paths.get_filename_list("loras"),),
                            "function_mode": (["none","tag", "clip","mask","infinite",],),
                            "select_method":("STRING",{ "default":""}),
                            "temperature": (
                                 "FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                             }}
    
    RETURN_TYPES = ("MODEL","CLIP")
    RETURN_NAMES = ("model","clip")
    FUNCTION = "easy_function_main"
    CATEGORY = "Storydiffusion"
    
    def easy_function_main(self, extra_repo,checkpoints,clip_vision,function_mode,select_method,temperature):
        use_gguf,use_unet=False,False
        model_path,clip,lora_path=None,None,None
    
        if checkpoints != "none":
            if checkpoints.endswith(".gguf"):
                model_path = folder_paths.get_full_path("gguf", checkpoints)
                use_gguf=True
            elif "glm" in checkpoints:
                model_path = folder_paths.get_full_path("clip", checkpoints)
            else:
                model_path = folder_paths.get_full_path("diffusion_models", checkpoints)
                use_unet=True

        if clip_vision != "none":
            if  "clip_vision" in clip_vision:
                clip_vision_path = folder_paths.get_full_path("clip_vision", clip_vision)     
            else:
                lora_path = folder_paths.get_full_path("loras", clip_vision)
                clip_vision_path = None
           
        else:
            clip_vision_path=None
        if function_mode=="tag":
            from .model_loader_utils import StoryLiteTag
            model = StoryLiteTag(device, temperature,select_method, extra_repo)
        elif function_mode=="clip":
            from .model_loader_utils import GLM_clip
            clip=GLM_clip(dir_path,model_path)
            model=None
        elif function_mode=="mask":
            model=None
        else:
            model=None
        
        use_svdq=True if "svdq" in select_method else False

        pipe={"model":model,"extra_repo":extra_repo,"model_path":model_path,"use_svdq":use_svdq,"use_gguf":use_gguf,"lora_ckpt_path":lora_path,
              "use_unet":use_unet,"extra_easy":function_mode,"select_method":select_method,"clip_vision_path":clip_vision_path}
        return (pipe,clip)


class StoryDiffusion_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "infer_mode": (["story", "classic","flux_pulid","infiniteyou","uno","story_maker","story_and_maker","consistory","kolor_face","msdiffusion" ],),
                    "photomake_ckpt": (["none"] + [i for i in folder_paths.get_filename_list("photomaker") if "v1" in i or "v2" in i],),
                    "ipadapter_ckpt": (["none"] + folder_paths.get_filename_list("photomaker"),),
                    "quantize_mode": ([ "fp8", "nf4","fp16", ],),
                    "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                    "extra_funtion":("STRING", {"default": ""}),
                            },
                "optional":{
                    "CLIP_VISION": ("CLIP_VISION",),
                            }
                }
    
    RETURN_TYPES = ("MODEL","DIFFCONDI",)
    RETURN_NAMES = ("model","switch",)
    FUNCTION = "main_apply"
    CATEGORY = "Storydiffusion"
    
    def main_apply(self,model,vae,infer_mode,photomake_ckpt,ipadapter_ckpt,quantize_mode,lora_scale,extra_funtion, **kwargs):

        
        photomake_ckpt_path = None if photomake_ckpt == "none"  else  folder_paths.get_full_path("photomaker", photomake_ckpt)
        ipadapter_ckpt_path = None if ipadapter_ckpt == "none"  else  folder_paths.get_full_path("photomaker", ipadapter_ckpt)
    
        CLIP_VISION=kwargs.get("CLIP_VISION")
        
        
        unet_type=torch.float16 #use for sdxl
        image_proj_model=None
        
        if isinstance(model,dict):
            cf_model_type=model.get("extra_easy")
            clip_vision_path=model.get("clip_vision_path")
            repo=model.get("extra_repo")
          
        else:
            cf_model_type=fitter_cf_model_type(model)
            clip_vision_path=None
            repo=None
           
        
        save_quantezed=True if "save" in extra_funtion and quantize_mode=="fp8" else False

        print(f"infer model type is {cf_model_type}")

        
        if clip_vision_path is None:    # 有2种加载clip vision的方式 
            clip_vision_path=CLIP_VISION if CLIP_VISION is not None else None
 
        if infer_mode=="msdiffusion" :
            if CLIP_VISION is None and clip_vision_path is None:
                raise "msdiffusion need a clipvison g model"
            elif CLIP_VISION is  None and clip_vision_path is not None:
                from comfy.clip_vision import load as clip_load
                CLIP_VISION=clip_load(clip_vision_path).model
        
        if infer_mode in ["story_maker" ,"story_and_maker"] and not CLIP_VISION  and ipadapter_ckpt_path is None:
             raise "story_maker need a clipvison H model,mask.bin"

        # check vram only using in flux pulid or UNO
        if total_vram > 45000.0:
            aggressive_offload = False
            offload = False
        elif 17000.0 < total_vram < 45000.0:
            aggressive_offload = False
            offload = True
        else:
            aggressive_offload = True
            offload = True

        logging.info(f"total_vram is {total_vram},aggressive_offload is {aggressive_offload},offload is {offload}")

        if infer_mode in["story", "story_and_maker","msdiffusion"]:# mix mode,use maker or ms to make 2 roles in on image
            model = Loader_storydiffusion(model,photomake_ckpt_path,vae)
        elif infer_mode =="story_maker":
            model = Loader_story_maker(model,ipadapter_ckpt_path,vae,False,lora_scale)
        elif infer_mode == "flux_pulid":
            from .PuLID.app_flux import get_models
            if isinstance(model,dict):
               ckpt_path=model.get("model_path")
               if ckpt_path is None:
                   raise "EasyFunction_Lite node must chocie a model"
            else:
                raise "PuLID can't link a normal comfyui model "
            
            model_=get_models("flux-dev",ckpt_path,False,aggressive_offload,device=device,offload=offload,quantized_mode=quantize_mode,)
            model = Loader_Flux_Pulid(model_,model,ipadapter_ckpt_path,quantize_mode,aggressive_offload,offload,False,clip_vision_path)
        elif infer_mode == "infiniteyou":
            model,image_proj_model = Loader_InfiniteYou(model,vae,quantize_mode)
        elif infer_mode == "consistory":
            model = load_pipeline_consistory(model,vae)
        elif infer_mode == "kolor_face":
            if not  isinstance(model,dict) :
                raise " must link EasyFunction_Lite node "
            else:
                if repo is None:
                    raise "EasyFunction_Lite node extra_repo must fill kolor repo"
            model = Loader_KOLOR(repo,clip_vision_path,ipadapter_ckpt_path)
        elif infer_mode == "uno":
            model = Loader_UNO(model,offload,quantize_mode,save_quantezed,lora_rank=512)
        else:  # can not choice a mode
            print("infer use comfyui classic mode")

        story_img=True if photomake_ckpt_path and infer_mode in["story","story_maker","story_and_maker","msdiffusion"] else False
        model_=model if infer_mode=="flux_pulid" or story_img else None

        return (model,{"infer_mode":infer_mode,"ipadapter_ckpt_path":ipadapter_ckpt_path,"photomake_ckpt_path":photomake_ckpt_path,"lora_scale":lora_scale,"image_proj_model":image_proj_model,
                       "CLIP_VISION":CLIP_VISION,"VAE":vae,"repo":repo,"model_":model_,"unet_type":unet_type,"extra_funtion":extra_funtion,"clip_vision_path":clip_vision_path})


        
class StoryDiffusion_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "switch": ("DIFFCONDI", {
                    "tooltip": "Switch infer mode witch your chocie."}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "display": "number"}),
                "role_text": ("STRING", {"multiline": True,"default": "[Taylor] a woman img, wearing a white T-shirt, blue loose hair.\n""[Lecun] a man img,wearing a suit,black hair."}),
                "scene_text":("STRING", {"multiline": True,
                                             "default": "[Taylor] wake up in the bed ;\n[Taylor] have breakfast by the window;\n[Lecun] driving a car;\n[Lecun] is working."}),
                "pos_text": ("STRING", {"multiline": True,"default": ",best"}),
                "neg_text": ("STRING", {"multiline": True,
                                               "default": "bad anatomy, bad hands, missing fingers, extra fingers,three hands, three legs, bad arms, missing legs, missing arms, poorly drawn "
                                                          "face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn,amputation, disconnected limbs"}),
                "lora_trigger_words": ("STRING", {"default": "best quality"}),
                "add_style": (["No_style", "Realistic", "Japanese_Anime", "Digital_Oil_Painting", "Pixar_Disney_Character","Photographic", "Comic_book","Line_art", "Black_and_White_Film_Noir", "Isometric_Rooms"],),
                "mask_threshold": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "extra_param":("STRING", {"default": ""}),
                "guidance_list": ("STRING", {"multiline": True, "default": "0., 0.25, 0.4, 0.75;0.6, 0.25, 1., 0.75"}),
            },
            "optional": {"tag_txt": ("CONDITIONING"),
                         "image":("IMAGE",),
                         "control_image":("IMAGE",),
                         }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","DIFFINFO","INT","INT",)
    RETURN_NAMES = ("positive", "negative","info","width","height",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "Storydiffusion"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip,switch,width,height, role_text,scene_text,pos_text,neg_text,lora_trigger_words,add_style,mask_threshold,extra_param,guidance_list,**kwargs):
        infer_mode=switch.get("infer_mode")
        use_lora=switch.get("use_lora")
        clip_vision = switch.get("CLIP_VISION")
        extra_funtion=switch.get("extra_funtion")
        photomake_ckpt_path=switch.get("photomake_ckpt_path")
        ipadapter_ckpt_path = switch.get("ipadapter_ckpt_path")
        unet_type=switch.get("unet_type")
        model_=switch.get("model_")
        vae=switch.get("VAE")
        
        
        tag_dict=kwargs.get("tag_txt")
        image=kwargs.get("image")
        control_image=kwargs.get("control_image")
        
        auraface,use_photov2,img2img_mode,cached,inject,onnx_provider=get_extra_function(extra_funtion,extra_param,photomake_ckpt_path,image,infer_mode)

        
        (replace_prompts,role_index_dict,invert_role_index_dict,ref_role_index_dict,ref_role_totals,role_list,role_dict,
         nc_txt_list,nc_indexs,positions_index_char_1,positions_index_char_2,positions_index_dual,prompts_dual,index_char_1_list,index_char_2_list)=pre_text2infer(role_text,scene_text,lora_trigger_words,use_lora,tag_dict)
        #print("role_index_dict:",replace_prompts,role_index_dict,invert_role_index_dict,ref_role_index_dict,ref_role_totals,role_list,role_dict,nc_txt_list,nc_indexs,positions_index_char_1,positions_index_char_2,positions_index_dual,prompts_dual,index_char_1_list,index_char_2_list)
        #role_index_dict: {'[Taylor]': [0, 1], '[Lecun]': [2, 3]} 
        # invert_role_index_dict{0: ['[Taylor]'], 1: ['[Taylor]'], 2: ['[Lecun]'], 3: ['[Lecun]']}
        # ref_role_index_dict {'[Taylor]': [0, 1], '[Lecun]': [2, 3]} 
        # ref_role_totals[0, 1, 2, 3]
        # role_list ['[Taylor]', '[Lecun]']
        # role_dict {'[Taylor]': ' a woman img, wearing a white T-shirt, blue loose hair.', '[Lecun]': ' a man img,wearing a suit,black hair.'} 
        # nc_txt_list [' a panda'] 
        # nc_indexs[4]
        # positions_index_dual[]
        # positions_index_char_1 0 2 
        # positions_index_char_2 [] 
        # prompts_dual[] #['[A] play whith [B] in the garden'] 
        # index_char_1_list [0, 1] 
        # index_char_2_list[2, 3]


        global character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals, character_dict
        character_index_dict=role_index_dict
        invert_character_index_dict=invert_role_index_dict
        ref_indexs_dict=ref_role_index_dict
        ref_totals=ref_role_totals
        character_dict=role_dict
        
        _, style_neg = apply_style_positive(add_style, " ") #get n
        neg_text = neg_text + style_neg
        
        replace_prompts=[i+pos_text for i in replace_prompts]
           # pre roles txt emb
        only_role_list=[apply_style_positive(add_style,i)[0] for i in replace_prompts]
        #print("only_role_list",only_role_list) ##w,m,w

        if len(role_list)==1:
            role_key_list=role_list*len(only_role_list)
        else:
            role_key_list=replicate_data_by_indices(role_list, index_char_1_list, index_char_2_list)
            role_key_list=[i for i in role_key_list if i is not None]

        if len(role_list)>1: #重新整理prompt排序，便于后续emb和ID的对应
            nc_dual_list=[]
            for i in nc_indexs:
                nc_dual_list.append(i)
            for i in positions_index_dual:
                nc_dual_list.append(i) 
            adjusted_a = adjust_indices(index_char_1_list, nc_dual_list)
            adjusted_b = adjust_indices(index_char_2_list, nc_dual_list)
            a_list = [only_role_list[i] for i in adjusted_a if 0 <= i < len(only_role_list)]
            b_list = [only_role_list[i] for i in adjusted_b if 0 <= i < len(only_role_list)]

            inf_list_split=[a_list,b_list]
        else:
            inf_list_split=[only_role_list]
        
        
        # if prompts_dual and infer_mode not in ["story_and_maker","story_maker","msdiffusion"]:
        #     raise "only support prompts_dual role in story maker and msdiffusion"

        if not img2img_mode and infer_mode=="flux_pulid":
            raise "flux_pulid mode only support image2image"

        if infer_mode=="msdiffusion" and not prompts_dual:
            raise "use msdiffusion mode need  have [role1] and [role2] in sence txt."
        
        if img2img_mode and   photomake_ckpt_path is None and infer_mode=="story":
            raise "need chocie photomake v1 or v2 when use img2img mode"
  
        if infer_mode=="story_maker"  and image is None:
            raise "story_maker only support  img2img mode,you can use story_and_maker or link a iamge"

        if infer_mode=="msdiffusion" and photomake_ckpt_path  and  not img2img_mode:
            raise "if use msdiffusion txt2img mode,can not use photomake_ckpt_path"

        uno_pe="d" #Literal['d', 'h', 'w', 'o']
        
        only_role_emb_ne,x_1_refs_dict,image_emb,x_1_refs_dual=None,None,None,None
        
        if img2img_mode:
            if infer_mode == "story_and_maker" or infer_mode == "story_maker":
                k1, _, _, _ = image.size()
                if k1 == 1:
                    image_emb = [clip_vision.encode_image(image)["penultimate_hidden_states"]]
                else:
                    img_list = list(torch.chunk(image, chunks=k1))
                    image_emb=[]
                    for i in img_list:
                        image_emb.append(clip_vision.encode_image(i)["penultimate_hidden_states"].to(device, dtype=unet_type))   
            elif infer_mode=="story":
                image_emb = None  # story的图片模式的emb要从ip的模型里单独拿，
            elif infer_mode=="msdiffusion":
                image_emb = clip_vision.encode_image(image)["penultimate_hidden_states"] #MS分情况，图生图直接在前面拿，文生图在在sample拿
                if not image_emb.is_cuda:#确保emb在cuda,以及type的正确
                    image_emb = image_emb.to(device,dtype=unet_type)
            elif infer_mode == "kolor_face":
                pass
            elif infer_mode == "flux_pulid":
                # get emb use insightface
                pass
            elif infer_mode == "infiniteyou":
                pass
            elif infer_mode == "uno":
                
                from .UNO.uno.flux.pipeline import preprocess_ref

                ref_pil_list=tensortopil_list(image) #原方法需要图片缩放，先转回pil
                if prompts_dual:
                    x_1_refs_dual_=[phi2narry(preprocess_ref(i, 320)) for i in ref_pil_list ]
                    x_1_refs_dual=[[vae.encode(i[:,:,:,:3]).to(torch.bfloat16) for i in x_1_refs_dual_]*len(prompts_dual)]#[[1,2],[1,2]] 不考虑cn
                
                if control_image is not None:
                    control_pil_list=tensortopil_list(control_image) 
                else:
                    control_pil_list=None

                if control_pil_list is None:# 无控制图时,默认时单图模式,x_1_refs用key取值
                    x_1_refs_dict={}    
                    for key ,role_pil,prompts in zip(role_list,ref_pil_list,inf_list_split):
                        role_tensor=phi2narry(preprocess_ref(role_pil, 512))#单图默认512
                        x_1_refs_dict[key]=[[vae.encode(role_tensor[:,:,:,:3]).to(torch.bfloat16)]*len(prompts)] # {key:[[1],[2],key2:[[3],[4]]}
                else:
                    if len(control_pil_list) != len(only_role_list):
                        raise "control image must same size as prompts,多少句话就多少张图"
                    else:
                        if len(role_list)>1:
                            nc_dual_list=[]
                            for i in nc_indexs:
                                nc_dual_list.append(i)
                            for i in positions_index_dual:
                                nc_dual_list.append(i) 
                            adjusted_a = adjust_indices(index_char_1_list, nc_dual_list)
                            adjusted_b = adjust_indices(index_char_2_list, nc_dual_list)
                            # print("adjusted_a",adjusted_a)
                            # print("adjusted_b",adjusted_b)
                            a_list = [control_pil_list[i] for i in adjusted_a if 0 <= i < len(control_pil_list)]
                            b_list = [control_pil_list[i] for i in adjusted_b if 0 <= i < len(control_pil_list)]

                            control_pil_list=[a_list,b_list]
                        else:
                            control_pil_list=[control_pil_list]
                        x_1_refs_dict={}
                        for key ,role_pil,control_pil in zip(role_list,ref_pil_list,control_pil_list):
                            mix_list=[]
                            role_tensor=phi2narry(preprocess_ref(role_pil, 320) )#多图默认320
                            for c in (control_pil):
                                c_tensor=phi2narry(preprocess_ref(c, 320))
                                mix_list.append([vae.encode(role_tensor[:,:,:,:3]).to(torch.bfloat16),vae.encode(c_tensor[:,:,:,:3]).to(torch.bfloat16)])
                            x_1_refs_dict[key]=mix_list #{key:[[1,2],[3,4]]}
            else:
                pass
        else:
            pass

        # pre insight face model and emb
        if img2img_mode:
            if infer_mode in ["story_and_maker","story_maker","flux_pulid","kolor_face","infiniteyou"] or (use_photov2 and infer_mode=="story"):

                from .model_loader_utils import insight_face_loader,get_insight_dict
                if switch.get("extra_repo") and switch.get("function_mode")=="mask":
                    mask_repo=switch.get("extra_repo")
                else:
                    mask_repo="briaai/RMBG-1.4"
                app_face,pipeline_mask,app_face_=insight_face_loader(infer_mode,use_photov2, auraface,onnx_provider,mask_repo)
                image_list=tensortopil_list_upscale(image, 640, 640)
                
                input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,input_id_cloth_dict=get_insight_dict(app_face,app_face_,pipeline_mask,infer_mode,use_photov2,image_list,
                     role_list,control_image,width, height,model_,switch.get("image_proj_model")) # CHECK role_list
                
            else:
                input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,input_id_cloth_dict={}, {}, {}, {}
                
        else:
            input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,input_id_cloth_dict={}, {}, {}, {}

        input_id_images_dict={}
        image_list=[]
        
        if img2img_mode and image is not None:
             if len(role_list)==1:
                image_pil=nomarl_upscale(image, width, height)
                input_id_images_dict[role_list[0]]=image_pil
                image_list=[image_pil]
             else:
                f1, _, _, _ = image.size()
                img_list = list(torch.chunk(image, chunks=f1))
                image_list = [nomarl_upscale(img, width, height) for img in img_list]
                for index, key in enumerate(role_list):
                    input_id_images_dict[key]=image_list[index]     

        #print("inf_list_split",inf_list_split)
        # pre clip txt emb
      
            
        noise_x=[]
        inp_neg_list=[]
        if  infer_mode=="consistory":
            only_role_emb=None
        elif infer_mode=="flux_pulid":
            from .PuLID.flux.util import load_clip, load_t5
            from .PuLID.app_flux  import get_emb_flux_pulid

            #repo_in="flux-dev" if not repo else repo
            if_repo =False
            t5_ = load_t5("flux-dev",clip,if_repo,device, max_length=128)
            clip_ = load_clip("flux-dev",clip,if_repo,device)
            only_role_emb,noise_x,inp_neg_list={},{},{}
            
            for key ,prompts in zip(role_list,inf_list_split):
                ip_emb,noise_,inp_n=[],[],[]
                
                for p,n in zip(prompts,[neg_text]*len(prompts)): 
                    seed_random = random.randint(0, MAX_SEED) #pulid 和uno的emb需要随机数
                    inp,inp_neg,x=get_emb_flux_pulid(t5_,clip_,if_repo,seed_random,p,n,width,height,num_steps=20,guidance=3.5,device=device)
                    ip_emb.append(inp)
                    inp_n.append(inp_neg)
                    noise_.append(x)
                only_role_emb[key]=ip_emb
                noise_x[key]=noise_
                inp_neg_list[key]=inp_n
        elif infer_mode == "uno":
            only_role_emb={}
            from .UNO.uno.flux.sampling import prepare_multi_ip_wrapper
            from .UNO.uno.flux.sampling import get_noise
            for key ,prompts in zip(role_list,inf_list_split):
                ip_emb=[]

                for p,x_1 in zip(prompts,x_1_refs_dict[key]):
                    seed_random = random.randint(0, MAX_SEED) #pulid 和uno的emb需要随机数
                    uno_x = get_noise(1, height, width, device=device,dtype=torch.bfloat16, seed=seed_random) 
                    inp = prepare_multi_ip_wrapper(clip,img=uno_x,prompt=p, ref_imgs=x_1, pe=uno_pe)
                    ip_emb.append(inp)
                only_role_emb[key]=ip_emb


        elif infer_mode=="kolor_face":
            from .kolors.models.tokenization_chatglm import ChatGLMTokenizer
            tokenizer = ChatGLMTokenizer.from_pretrained(os.path.join(switch.get("repo"),'text_encoder'))

            chatglm3_model = {
                'text_encoder': clip, 
                'tokenizer': tokenizer
                }
            only_role_emb,only_role_emb_ne=glm_single_encode(chatglm3_model, inf_list_split,role_list, neg_text, 1) 
        else:
            if photomake_ckpt_path is not None and img2img_mode and infer_mode in["story","story_maker","story_and_maker","msdiffusion"]: #img2img模式下SDXL的story的clip要特殊处理，有2个imgencoder进程，所以分离出来 TODO
                if use_photov2:
                    if len(role_list)==1:
                        emb_dict=photomaker_clip_v2(clip,model_,only_role_list,neg_text,image_list,input_id_emb_s_dict[role_key_list[0]][0])
                        only_role_emb=[emb_dict]
                    else:
                        
                        only_role_emb=[]
                        for role_list_s,role_list_id,key in zip(inf_list_split,image_list,role_key_list):
                            #print(input_id_emb_s_dict[key][0].shape) #torch.Size([512])
                            emb_dict=photomaker_clip_v2(clip,model_,role_list_s,neg_text,[role_list_id],input_id_emb_s_dict[key][0])
                            only_role_emb.append(emb_dict)
                  
                else: 
                    if len(role_list)==1:
                        emb_dict=photomaker_clip(clip,model_,only_role_list,neg_text,image_list)
                        only_role_emb=[emb_dict]
                    else:
                    
                        only_role_emb=[]
                        for role_list_s,role_list_id in zip(inf_list_split,image_list):
                            emb_dict=photomaker_clip(clip,model_,role_list_s,neg_text,[role_list_id])
                            only_role_emb.append(emb_dict)
                    
            else:
                if infer_mode=="classic": #TODO 逆序的角色会出现iD不匹配，受影响的有story文生图
                    only_role_emb= cf_clip(only_role_list, clip, infer_mode,role_list) #story模式需要拆分prompt，所以这里需要传入role_list
                else:
                    only_role_emb= cf_clip(inf_list_split, clip, infer_mode,role_list)  #story,story_maker,story_and_maker,msdiffusion,infinite
        # pre nc txt emb
        if nc_txt_list and not infer_mode=="consistory":
            nc_txt_list=[i+pos_text for i in nc_txt_list]
            if photomake_ckpt_path is not None and img2img_mode and infer_mode in["story","story_maker","story_and_maker","msdiffusion"]: #img2img模式下SDXL的story的clip要特殊处理，有2个imgencoder进程，所以分离出来 TODO
                nc_emb=[]
                for  i  in nc_txt_list:
                    if use_photov2:
                        empty_emb_zero = torch.zeros_like(input_id_emb_s_dict[role_list[0]][0]).to(device,dtype=torch.float16)
                        emb_dict_=photomaker_clip_v2(clip,model_,[i],neg_text,image_list,empty_emb_zero,nc_flag=True)
                    else:
                        emb_dict_=photomaker_clip(clip,model_,[i],neg_text,image_list,nc_flag=True)
                    nc_emb.append(emb_dict_)
            else:
                if infer_mode!="kolor_face":
                    nc_emb=cf_clip(nc_txt_list, clip, infer_mode,role_list,input_split=False)
                else:
                    nc_emb,_= glm_single_encode(chatglm3_model, nc_txt_list,role_list, neg_text, 1,nc_mode=True) 
        else:
            nc_emb=None
        # pre dual role txt emb
        grounding_kwargs=None
        cross_attention_kwargs=None
        if prompts_dual and infer_mode in["story_maker","story_and_maker","msdiffusion"] : #忽略不支持的模式
    
            if infer_mode=="msdiffusion": #[A] a (pig) play whith [B]  a (doll) in the garden
                prompts_dual=[i.replace(role_list[0] ,role_dict[role_list[0]]) for i in prompts_dual if role_list[0] in i ]
                prompts_dual = [i.replace(role_list[1], role_dict[role_list[1]]) for i in prompts_dual if role_list[1] in i]
                if use_lora:
                    prompts_dual=[i+lora_trigger_words for i in prompts_dual]
                prompts_dual=[apply_style_positive(add_style,i+pos_text)[0] for i in prompts_dual] #[' T a (pig)  play whith  a (doll) in the garden,best 8k,RAW']

                if '(' in prompts_dual[0] and ')' in prompts_dual[0]:
                    object_prompt = extract_content_from_brackets_(pos_text)  # 提取prompt的object list
                    object_prompt=[i.strip() for i in object_prompt]
                    for i in object_prompt:
                        if " " in i:
                            raise "when using [object],object must be a word,any blank in it will cause error."
                        
                    object_prompt=[i for i in object_prompt ]
                    phrases = sorted(list(set(object_prompt)),key=lambda x: list(object_prompt).index(x))  # 清除同名物体,保持原有顺序
                    assert  len(phrases)>= 2
                    if len(phrases)>2:
                        phrases=phrases[:2] #只取前两个物体
                else:
                    raise "when using msdiffusion ,(objectA)  and (objectA) must be in the prompt."


                prompts_dual=[i.replace("("," ").replace(")"," ") for i in prompts_dual] #clear the bracket

                box_add = []  # 获取预设box
                guidance_list = guidance_list.strip().split(";")
                for i in range(len(guidance_list)):
                    box_add.append(get_float(guidance_list[i]))
                
                if mask_threshold == 0:
                    mask_threshold = None
                if mask_threshold:
                    boxes = [box_add[:2]]  # boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # man+women
                else:
                    boxes = [[[0, 0, 0, 0], [0, 0, 0, 0]]]  # used if you want no layout guidance
                print(f"Roles position on {boxes}")
                from transformers import CLIPTokenizer
                tokenizer_=CLIPTokenizer.from_pretrained(os.path.join(dir_path, "local_repo/tokenizer"))
                for i in prompts_dual:
                    phrase_idxes = [get_phrases_idx_cf(tokenizer_, phrases[0], i)]
                    eot_idxes = [[get_eot_idx_cf(tokenizer_, i)] * len(phrases[0])]
                    cross_attention_kwargs, grounding_kwargs = get_ms_phrase_emb(boxes, device, infer_type_g,
                                                                             [0], 1, phrase_idxes,
                                                                             1, eot_idxes, phrases, clip,tokenizer_)
                daul_emb = cf_clip(prompts_dual, clip, infer_mode,role_list,input_split=False)
            else:
                prompts_dual=[i.replace(role_list[0] ,role_dict[role_list[0]]) for i in prompts_dual if role_list[0] in i ]
                prompts_dual = [i.replace(role_list[1], role_dict[role_list[1]]) for i in prompts_dual if role_list[1] in i]
                if use_lora:
                    prompts_dual=[i+lora_trigger_words for i in prompts_dual]
                prompts_dual=[apply_style_positive(add_style,i+pos_text)[0] for i in prompts_dual] #[' The figurine  play whith  The pig in the garden,best 8k,RAW']
                daul_emb=cf_clip(prompts_dual, clip, infer_mode,role_list,input_split=False) # maker
        elif prompts_dual and infer_mode == "uno": #UNO双角色图片要单独处理
            prompts_dual=[i.replace(role_list[0] ,role_dict[role_list[0]]) for i in prompts_dual if role_list[0] in i ]
            prompts_dual = [i.replace(role_list[1], role_dict[role_list[1]]) for i in prompts_dual if role_list[1] in i]
            prompts_dual=[apply_style_positive(add_style,i+pos_text)[0] for i in prompts_dual] #[' The figurine  play whith  The pig in the garden,best 8k,RAW']
           
            daul_emb=[]
            for dual_t,x_1 in zip(prompts_dual,x_1_refs_dual): # dual_t:The figurine  play whith  The pig in the garden best 8k,RAW
                seed_random = random.randint(0, MAX_SEED) 
                uno_x = get_noise(1, height, width, device=device,dtype=torch.bfloat16, seed=seed_random) 
                inp = prepare_multi_ip_wrapper(clip,img=uno_x,prompt=dual_t, ref_imgs=x_1, pe=uno_pe)
                daul_emb.append(inp)
        else:
            daul_emb=None
        # neg
        if infer_mode=="consistory":
            negative = None
            postive_dict={}
        elif infer_mode=="flux_pulid":
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb} #不支持NC
            negative = [inp_neg_list,noise_x]
        elif infer_mode=="uno":
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb} #TODO
            negative=None
        elif infer_mode=="kolor_face":
            postive_dict = {"role": only_role_emb, "nc": nc_emb, "daul": daul_emb}
            negative = only_role_emb_ne[0]
        else:
            tokens_n = clip.tokenize(neg_text)
            output_n = clip.encode_from_tokens(tokens_n, return_pooled=True, return_dict=True) #{"pooled_output":tensor}
            cond_n = output_n.pop("cond")
            if cond_n.shape[1] /77>1 and infer_mode != "classic":
                logging.warning("nagetive prompt'tokens length is abvoe 77,will split it.")
                cond_n=torch.chunk(cond_n,cond_n.shape[1] //77,dim=1)[0]
        
            if infer_mode == "classic":
                if nc_emb is not None:
                    for index,i in zip(nc_indexs,nc_emb):
                        only_role_emb.insert(index,i)
                if  daul_emb is not None:
                    for index,i in zip(positions_index_dual,daul_emb):
                        only_role_emb.insert(index,i)
                postive_dict = only_role_emb
                negative = [[cond_n, output_n]]
            else:
                postive_dict = {"role": only_role_emb, "nc": nc_emb, "daul": daul_emb}
                negative = [cond_n, output_n]
        
            

        # Pre emb for maker,
        
        if img2img_mode and infer_mode in ["story_and_maker","story_maker"]:
            num_images_per_prompt=1
            make_img,make_mask_img,make_face_info,make_cloth_info=[],[],[],[]
            
            for key in role_list:
                img_ = input_id_emb_un_dict[key][0]
                # print(character_key_str,input_id_images_dict)
                mask_image_ = input_id_img_s_dict[key][0] #mask_image
                face_info_ = input_id_emb_s_dict[key][0]
                cloth_info_ = None
                if isinstance(control_image, torch.Tensor):
                    cloth_info_ = input_id_cloth_dict[key][0]
                make_img.append(img_)
                make_mask_img.append(mask_image_)
                make_face_info.append(face_info_)
                make_cloth_info.append(cloth_info_)
            
            mask_image_2=None
            face_info_2=None
            cloth_2=None
            image_2=None
            prompt_image_emb_dual=None
            if len(role_list)>1:
               
                if isinstance(control_image, torch.Tensor):
                    cn_image_list=tensortopil_list_upscale(control_image,width,height)
                    if len(cn_image_list)<len(only_role_list): #防止溢出
                        cn_image_list=cn_image_list + cn_image_list[0]*len(only_role_list)-len(cn_image_list)
                    nc_dual_list=[]
                    for i in nc_indexs:
                        nc_dual_list.append(i)
                    for i in positions_index_dual:
                        nc_dual_list.append(i) 
                    
                    adjusted_a_ = adjust_indices(index_char_1_list, nc_dual_list)
                    adjusted_b_ = adjust_indices(index_char_2_list, nc_dual_list)
                    a_list = [cn_image_list[i] for i in adjusted_a_ if 0 <= i < len(cn_image_list)]
                    b_list = [cn_image_list[i] for i in adjusted_b_ if 0 <= i < len(cn_image_list)]
                    
                    prompt_image_emb,maker_control_image={},{}
                    for j,(key,cnlist) in enumerate(zip(role_list,[a_list,b_list])):
                        p_list,maker_cn_list=[],[]

                        for cn_img in cnlist:
                            prompt_image_emb_,maker_control_image_=encode_prompt_image_emb_([make_img[j],cn_img[i]], a_list, make_mask_img[j], b_list, make_face_info[j], face_info_2, make_cloth_info[j],
                                                                                             cloth_2,device, num_images_per_prompt, unet_type, clip_vision,vae,do_classifier_free_guidance=True)
                            p_list.append(prompt_image_emb_)
                            maker_cn_list.append(maker_control_image_)
                        prompt_image_emb[key]=p_list#输出改为字典
                        maker_control_image[key]=maker_cn_list#输出改为字典
                    # 还要将每个元素转为列表，改为[img,cn_img]

                else:
                    prompt_image_emb,maker_control_image={},{}
                    
                    for i,key in enumerate(role_list):
                        prompt_image_emb_,maker_control_image_=encode_prompt_image_emb_(make_img[i], image_2, make_mask_img[i], mask_image_2, make_face_info[i], face_info_2, make_cloth_info[i], cloth_2,device, num_images_per_prompt, unet_type, clip_vision,vae,do_classifier_free_guidance=True)
                        prompt_image_emb[key]=[prompt_image_emb_]#输出改为字典
                        maker_control_image[key]=[maker_control_image_]#输出改为字典
      
            else:
                prompt_image_emb,maker_control_image=encode_prompt_image_emb_(make_img[0], image_2, make_mask_img[0], mask_image_2, make_face_info[0], face_info_2, make_cloth_info[0], cloth_2,device, num_images_per_prompt, unet_type, clip_vision,vae,do_classifier_free_guidance=True)
                prompt_image_emb=={role_list[0]:len(only_role_list)*[prompt_image_emb]}#输出改为字典
                maker_control_image={role_list[0]:len(only_role_list)*[maker_control_image]}#输出改为字典

            maker_control_image_dual=None
            if daul_emb is not None:
                prompt_image_emb_dual,maker_control_image_dual=encode_prompt_image_emb_(make_img[0], make_img[1], make_mask_img[0], make_mask_img[1], make_face_info[0], make_face_info[1], make_cloth_info[0], make_cloth_info[1],device, num_images_per_prompt, unet_type, clip_vision,vae,do_classifier_free_guidance=True)
                prompt_image_emb_dual=len(prompts_dual)*[prompt_image_emb_dual]
                maker_control_image_dual=len(prompts_dual)*[maker_control_image_dual] 
        else:   
            prompt_image_emb,maker_control_image,prompt_image_emb_dual,maker_control_image_dual=None,None,None,None

              
        
        # switch
        switch["id_len"]=len(role_list)
        switch["role_list"] = role_list
        switch["invert_role_index_dict"]=invert_role_index_dict
        switch["nc_index"] = nc_indexs
        switch["dual_index"] = positions_index_dual
        switch["grounding_kwargs"]=grounding_kwargs
        switch["cross_attention_kwargs"]=cross_attention_kwargs
        switch["image_embeds"] = image_emb
        switch["img2img_mode"] =img2img_mode
        switch["positions_index_char_1"] =positions_index_char_1
        switch["positions_index_char_2"] =positions_index_char_2
        switch["mask_threshold"] =mask_threshold
        switch["clip_vision"]=clip_vision
        switch["input_id_emb_s_dict"]=input_id_emb_s_dict
        switch["input_id_img_s_dict"]=input_id_img_s_dict
        switch["input_id_emb_un_dict"]=input_id_emb_un_dict
        switch["maker_control_image"]=maker_control_image
        switch["prompt_image_emb"]=prompt_image_emb
        switch["prompt_image_emb_dual"]=prompt_image_emb_dual
        switch["maker_control_image_dual"]=maker_control_image_dual
        switch["prompts_dual"]= prompts_dual
        switch["control_image"] = control_image
        switch["only_role_list"]=only_role_list
        switch["nc_txt_list"]=nc_txt_list
        switch["neg_text"]=neg_text
        switch["role_text"]=role_text
        switch["cf_clip"] =clip
        switch["cached"]=cached
        switch["inject"]=inject
        switch["role_key_list"]=role_key_list
        switch["input_id_images_dict"]=input_id_images_dict
        switch["id_index"]=(index_char_1_list,index_char_2_list)
        
 
        return (postive_dict,negative,switch,width,height)
        

class StoryDiffusion_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "info":("DIFFINFO",{
                    "tooltip": "Switch infer mode witch your chocie."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "sa32_degree": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "sa64_degree": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            },
                
        }
    
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    
    CATEGORY = "Storydiffusion"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."
    
    def common_ksampler(self,model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                        disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent_image,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                      last_step=last_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                      disable_pbar=disable_pbar, seed=seed)
        # out = latent.copy()
        # out["samples"] = samples
        return samples
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,info, latent_image,sa32_degree,sa64_degree, denoise=1.0, **kwargs):
       
        infer_mode=info.get("infer_mode")
        id_len=info.get("id_len")
        nc_index=info.get("nc_index")
        dual_index=info.get("dual_index")
        role_list=info.get("role_list")
        mask_threshold=info.get("mask_threshold")
        grounding_kwargs=info.get("grounding_kwargs")
        cross_attention_kwargs=info.get("cross_attention_kwargs")
        image_embeds=info.get("image_embeds")
        img2img_mode=info.get("img2img_mode")
        num_images_per_prompt=1
        photomake_ckpt_path = info.get("photomake_ckpt_path")
        ipadapter_ckpt_path = info.get("ipadapter_ckpt_path")
        prompt_image_emb=info.get("prompt_image_emb")
        maker_control_image=info.get("maker_control_image")
        prompt_image_emb_dual=info.get("prompt_image_emb_dual")
        maker_control_image_dual=info.get("maker_control_image_dual")
        controlnet=info.get("controlnet")
        prompts_dual=info.get("prompts_dual")
        input_id_emb_un_dict=info.get("input_id_emb_un_dict")
        input_id_img_s_dict=info.get("input_id_img_s_dict")
        input_id_emb_s_dict=info.get("input_id_emb_s_dict")
        input_id_cloth_dict=info.get("input_id_cloth_dict")
        only_role_list=info.get("only_role_list")
        nc_txt_list=info.get("nc_txt_list")
        neg_text=info.get("neg_text")
        cached=info.get("cached")
        inject=info.get("inject")
        input_id_images_dict=info.get("input_id_images_dict")
        
        invert_role_index_dict=info.get("invert_role_index_dict")
        latent_init=latent_image["samples"]
        scheduler_choice = get_scheduler(sampler_name, scheduler)
        #from latent get h & w
        batch_size,_,height, width = latent_init.size()
        height=height*8
        width=width*8
        empty_img_init = Image.new('RGB', (height, width), (255, 255, 255))
        zero_tensor =  torch.zeros((height, width), dtype=torch.float32, device="cpu")
        
        control_image_zero = Image.fromarray(np.zeros([height, width, 3]).astype(np.uint8))
        start_merge_step = int(float(20) / 100 * steps) # TODO: 20% of steps
        if start_merge_step > 30:
            start_merge_step = 30

        if  infer_mode =="classic":
            samples_list=[]
            for i in positive:
                seed_random=random.randint(0, seed)
                samples=self.common_ksampler(model, seed_random, steps, cfg, sampler_name, scheduler, i, negative, latent_image,
                                denoise=denoise)  #torch.Size([1, 4, 64, 64])
               
                samples_list.append(samples)
            out = latent_image.copy()
            out["samples"] = torch.cat(samples_list,dim=0)
            return (out,)
        else:
            #get emb
            only_role_emb=positive.get("role")
            nc_emb=positive.get("nc")
            daul_emb=positive.get("daul")
            daul_emb_ms=daul_emb
            
            if infer_mode in["story" ,"msdiffusion","story_and_maker"]: #三者都调用story的unet方法，只是双角色引入ms或者maker
                if ipadapter_ckpt_path is None and infer_mode=="msdiffusion":
                    raise "msdiffusion  need a ms_adapter.bin file at ipadapter_ckpt menu."
                global attn_procs,sa32, sa64, write, height_s, width_s,attn_count, total_count, id_length, total_length, cur_step,cur_character

                sa32 = sa32_degree
                sa64 = sa64_degree
                attn_count = 0
                total_count = 0
                cur_step = 0
                id_length = id_len
                total_length = 5
                attn_procs = {}
                write = False
                height_s = height
                width_s = width
                
                set_attention_processor(model.unet, id_len, is_ipadapter=False)
                
                model.scheduler = scheduler_choice.from_config(model.scheduler.config)
                model.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
                
                samples_list = []
                if img2img_mode and photomake_ckpt_path:
                    generator=torch.Generator(device=device).manual_seed(seed)
                    #print(invert_role_index_dict)
                    for emb_dict,r_key,key in zip(only_role_emb,role_list,invert_role_index_dict.keys()):
                        write = True
                        cur_character = invert_role_index_dict[key] #全局角色名称，需要
                        #print(input_id_images_dict) #{'[Taylor]': <PIL.Image>, '[Lecun]': <PIL.Image>}
                        #print(r_key)# '[Taylor]'
                        # print(emb_dict)
                        samples = model(
                                input_id_images=input_id_images_dict[r_key],
                                num_inference_steps=steps,
                                guidance_scale=cfg,
                                start_merge_step=start_merge_step,
                                height=height,
                                width=width,
                                negative_prompt=neg_text,
                                generator=generator,
                                emb_dict=emb_dict,
                                )[0]  # torch.Size([1, 4, 64, 64])
                        #print(samples.shape)
                        samples_list.append(samples)
                    if nc_emb: #no role 无角色的emb
                        nc_list=[]
                        for i in (nc_emb):
                            write = False
                            samples = model(input_id_images=[input_id_images_dict[role_list[0]]],
                                    num_inference_steps=steps,
                                    guidance_scale=cfg,
                                    start_merge_step=start_merge_step,
                                    height=height,
                                    width=width,
                                    negative_prompt=neg_text,
                                    generator=generator,
                                    emb_dict=i,
                                    nc_flag=True)[0]  # torch.Size([1, 4, 64, 64])
                            nc_list.append(samples)
                        for index_,sample_nc in zip(nc_index,nc_list):
                            samples_list.insert(int(index_), sample_nc)
                else:
                    for key in role_list: # i:{a:[[tensor,tensor],[tensor,tensor]],b:[[tensor,tensor],[tensor,tensor]]} 只跑第一张图生成ID
                        seed_random = random.randint(0, seed)
                        write = True
                        cur_character = [key]
                        samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                                        generator=torch.Generator(device=device).manual_seed(seed_random),
                                        prompt_embeds=only_role_emb[key][0][0],
                                        negative_prompt_embeds=negative[0],
                                        pooled_prompt_embeds=only_role_emb[key][0][1].get("pooled_output"),
                                        negative_pooled_prompt_embeds=negative[1].get("pooled_output"))[
                            0]  # torch.Size([1, 4, 64, 64])
                        samples_list.append(samples)
                       
                        write = False #关闭角色生成
                    
                        for index,emb in enumerate(only_role_emb[key]):
                            if index==0:
                                continue
                            samples_ = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                                                generator=torch.Generator(device=device).manual_seed(seed_random),
                                                prompt_embeds=emb[0],
                                                negative_prompt_embeds=negative[0],
                                                pooled_prompt_embeds=emb[1].get("pooled_output"),
                                                negative_pooled_prompt_embeds=negative[1].get("pooled_output"))[
                                    0]  # torch.Size([1, 4, 64, 64])
                            samples_list.append(samples_)
                                
                    if nc_emb: #no role 无角色的emb
                        for index, i in zip(nc_index, nc_emb):
                            seed_random = random.randint(0, seed)
                            write = False
                            samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                                            generator=torch.Generator(device=device).manual_seed(seed_random),
                                            prompt_embeds=i[0],
                                            negative_prompt_embeds=negative[0],
                                            pooled_prompt_embeds=i[1].get("pooled_output"),
                                            negative_pooled_prompt_embeds=negative[1].get("pooled_output"))[
                                0]  # torch.Size([1, 4, 64, 64])
                            samples_list.insert(index, samples)
                if infer_mode=="msdiffusion" or infer_mode =="story_and_maker":
                    daul_emb = None #确保pass后段dual的执行
                    if daul_emb_ms:
                        # del model
                        # gc.collect()
                        # torch.cuda.empty_cache()
                       
                        VAE=info.get("VAE")
                        for j ,(index, i) in enumerate(zip(dual_index, daul_emb_ms)):
                            seed_random = random.randint(0, seed)
                            write = False
                            if not img2img_mode: #文生图模式，以文生图第一张为ID参考拿emb
                                clip_vision = info.get("CLIP_VISION")
                                
                                out_1, out_2 = {}, {}
                                out_1["samples"]=samples_list[int(info.get("positions_index_char_1"))]
                                out_2["samples"] =samples_list[int(info.get("positions_index_char_2"))]
                                role_1=VAE.decode(out_1["samples"])
                                role_2 = VAE.decode(out_2["samples"])
                               
                                if  infer_mode=="msdiffusion":
                                    role_tensor=torch.cat((role_1,role_2),dim=0)
                                    image_embeds=clip_vision.encode_image(role_tensor)["penultimate_hidden_states"]
                                    image_embeds = image_embeds.to(device, dtype=model.unet.dtype)
                                    if controlnet:
                                        model=model #TO DO
                                    #write = False
                                    samples = Infer_MSdiffusion(model,ipadapter_ckpt_path,image_embeds, i[0],negative[0],grounding_kwargs,cross_attention_kwargs,
                                                                 1,mask_threshold,height,width,steps,seed_random,cfg,i[1].get("pooled_output"),negative[1].get("pooled_output") )

                                else: #maker的emb要单独处理，用列表形式传入 make
                                    # image_embeds=[]
                                    # for X in [role_1,role_2]:
                                    #     image_embeds.append(clip_vision.encode_image(X)["penultimate_hidden_states"].to(device, dtype=model.unet.dtype))
                                    make_img,make_mask_img,make_face_info,make_cloth_info=[],[],[],[]
                                    from .model_loader_utils import insight_face_loader,get_insight_dict
                                    app_face,pipeline_mask,app_face_=insight_face_loader(infer_mode,False, False)
                                   
                                    image_list = [tensor_to_image(role_1),tensor_to_image(role_2)]
                                    input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,input_id_cloth_dict=get_insight_dict(app_face,app_face_,pipeline_mask,infer_mode,False,image_list,role_list,info.get("control_image"),width, height) 
                                    for key in role_list:
                                        img_ = input_id_emb_un_dict[key][0]
                                        # print(character_key_str,input_id_images_dict)
                                        mask_image_ = input_id_img_s_dict[key][0] #mask_image
                                        face_info_ = input_id_emb_s_dict[key][0]
                                        cloth_info_ = None
                                        if isinstance(info.get("control_image"), torch.Tensor):
                                            cloth_info_ = input_id_cloth_dict[key][0]
                                        make_img.append(img_)
                                        make_mask_img.append(mask_image_)
                                        make_face_info.append(face_info_)
                                        make_cloth_info.append(cloth_info_)

                                    prompt_image_emb_dual,maker_control_image_dual=encode_prompt_image_emb_(make_img[0], make_img[1], make_mask_img[0], make_mask_img[1], make_face_info[0],
                                                                                                             make_face_info[1], make_cloth_info[0], make_cloth_info[1],device, num_images_per_prompt, info.get("unet_type"), clip_vision,VAE,do_classifier_free_guidance=True)
                                    prompt_image_emb_dual=len(prompts_dual)*[prompt_image_emb_dual]
                                    maker_control_image_dual=len(prompts_dual)*[maker_control_image_dual]
                                    samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                                        generator=torch.Generator(device=device).manual_seed(seed_random),
                                        prompt_embeds=i[0],
                                        negative_prompt_embeds=negative[0],
                                        pooled_prompt_embeds=i[1].get("pooled_output"),
                                        negative_pooled_prompt_embeds=negative[1].get("pooled_output"),
                                        prompt_image_emb=prompt_image_emb_dual[j],
                                        control_image=maker_control_image_dual[j],
                                        )[0]  # torch.Size([1, 4, 64, 64])
                            else:
                                if  infer_mode=="msdiffusion":
                                    #del model
                                    model=Loader_storydiffusion(None,None,None,model)
                                    gc_cleanup()
                                    samples = Infer_MSdiffusion(model,ipadapter_ckpt_path,image_embeds, i[0],negative[0],grounding_kwargs,cross_attention_kwargs,
                                                                 1,mask_threshold,height,width,steps,seed_random,cfg,i[1].get("pooled_output"),negative[1].get("pooled_output") )
                                #del model
                                else:
                                    print("reload maker") 
                                    model=Loader_story_maker(None,ipadapter_ckpt_path,VAE,False,info.get("lora_scale"),UNET=model)
                                    gc_cleanup()
                                    if controlnet:
                                        model.controlnet = controlnet
                                        
                                    seed_random = random.randint(0, seed)
                                   
                                    #print(i)
                                    samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                                        generator=torch.Generator(device=device).manual_seed(seed_random),
                                        prompt_embeds=i[0],
                                        negative_prompt_embeds=negative[0],
                                        pooled_prompt_embeds=i[1].get("pooled_output"),
                                        negative_pooled_prompt_embeds=negative[1].get("pooled_output"),
                                        prompt_image_emb=prompt_image_emb_dual[j],
                                        control_image=maker_control_image_dual[j],
                                        )[0]  # torch.Size([1, 4, 64, 64])
                            samples_list.insert(index,samples)
            
                if daul_emb: #双角色的emb，即便不是msdiffusion，其他方法也能用，只是ID不一致而已
                    for index, i in zip(dual_index, daul_emb):
                        seed_random = random.randint(0, seed)
                        write = False
                        samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                            generator=torch.Generator(device=device).manual_seed(seed_random),
                            prompt_embeds=i[0],
                            negative_prompt_embeds=negative[0],
                            pooled_prompt_embeds=i[1].get("pooled_output"),
                            negative_pooled_prompt_embeds=negative[1].get("pooled_output"),
                            )[0]  # torch.Size([1, 4, 64, 64])
                        samples_list.insert(index, samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
            elif infer_mode == "story_maker" : #单纯使用maker，兼容单体双人及双人同框，目前需要修改源码，将cn的图片与衣服的emb拿出来提前处理，再传入pipe
                model.scheduler = scheduler_choice.from_config(model.scheduler.config)
                seed_random = random.randint(0, seed)
                samples_list = []
                for key in role_list :
                    for i,emb in enumerate(only_role_emb[key]): 
                        samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                            generator=torch.Generator(device=device).manual_seed(seed_random),
                            prompt_embeds=emb[0],
                            negative_prompt_embeds=negative[0],
                            pooled_prompt_embeds=emb[1].get("pooled_output"),
                            negative_pooled_prompt_embeds=negative[1].get("pooled_output"),
                            prompt_image_emb=prompt_image_emb[key][i],
                            control_image=maker_control_image[key][i],
                            )[0]  # torch.Size([1, 4, 64, 64])
                        samples_list.append(samples)
                if nc_emb: #no role 无角色的emb
                    id_embeds_z, clip_image_embeds_z, clip_face_embeds_z = torch.zeros_like(prompt_image_emb[0]), torch.zeros_like(prompt_image_emb[1]), torch.zeros_like(prompt_image_emb[2])
                    for index, i in zip(nc_index, nc_emb):
                        seed_random = random.randint(0, seed)
                        write = False
                        samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                            generator=torch.Generator(device=device).manual_seed(seed_random),
                            prompt_embeds=i[0],
                            negative_prompt_embeds=negative[0],
                            pooled_prompt_embeds=i[1].get("pooled_output"),
                            negative_pooled_prompt_embeds=negative[1].get("pooled_output"),
                            prompt_image_emb=(id_embeds_z, clip_image_embeds_z, clip_face_embeds_z),
                            control_image=empty_img_init,
                            )[0]  # torch.Size([1, 4, 64, 64])
                        samples_list.insert(index, samples)
                if daul_emb:
                    for j ,(index, i) in enumerate(zip(dual_index, daul_emb)):
                        seed_random = random.randint(0, seed)
                        write = False
                        samples = model(height=height, width=width, num_inference_steps=steps, guidance_scale=cfg,
                            generator=torch.Generator(device=device).manual_seed(seed_random),
                            prompt_embeds=i[0],
                            negative_prompt_embeds=negative[0],
                            pooled_prompt_embeds=i[1].get("pooled_output"),
                            negative_pooled_prompt_embeds=negative[1].get("pooled_output"),
                            prompt_image_emb=prompt_image_emb_dual[j],
                            control_image=maker_control_image_dual[j],
                            )[0]  # torch.Size([1, 4, 64, 64])
                        samples_list.insert(index, samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)

            elif infer_mode == "consistory":
                from .consistory.consistory_run import run_batch_generation,run_anchor_generation, run_extra_generation
                mask_dropout = 0.5
                same_latent = False
                n_achors = 2
                role_input=info.get("role_text").splitlines()[0]
                main_role = role_input.replace("]", "").replace("[", "")
                concept_token = [main_role] 
                if ")" in role_input:
                   object_role = role_input.split(")")[0].split("(")[-1]
                   concept_token=[main_role,object_role]
                style = "A photo of "
                subject=f"a {main_role} "
                replace_prompts= [f'{style}{subject} {i}' for i in only_role_list]
                gpu = 0
                torch.cuda.reset_max_memory_allocated(gpu)
                
                model.scheduler = scheduler_choice.from_config(model.scheduler.config)
                seed_random = random.randint(0, seed)
                if not cached:
                    if not inject:
                        model.enable_vae_slicing()
                        model.enable_model_cpu_offload()
                    else:
                        model.to(torch.float16)
                    samples_list = run_batch_generation(model, replace_prompts, concept_token,neg_text, seed,n_steps=steps,mask_dropout=mask_dropout,
                                                         same_latent=same_latent, perform_injection=inject,n_achors=n_achors,cf_clip=info.get("cf_clip"))
                    out = {}
                    out["samples"] = samples_list
                    return (out,zero_tensor)
                else:  
                    samples_list = []
                    if len(replace_prompts)>2:
                        spilit_prompt=replace_prompts[:2]
                    else:
                        spilit_prompt=replace_prompts
                        
                    anchor_out_images, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(
                        model, spilit_prompt, concept_token,neg_text,
                        seed=seed, n_steps=steps, mask_dropout=mask_dropout, same_latent=same_latent,perform_injection=inject,
                        cache_cpu_offloading=True,cf_clip=info.get("cf_clip"))
                    samples_list.append(anchor_out_images)
                    if len(replace_prompts) > 2:
                        left_prompt=replace_prompts[2:]
                    else:
                        left_prompt=replace_prompts[:1]  # use default
                    
                    for extra_prompt in left_prompt:
                        extra_image = run_extra_generation(model, [extra_prompt], concept_token,neg_text,
                            anchor_cache_first_stage,
                            anchor_cache_second_stage,
                            seed=seed, n_steps=steps,
                            mask_dropout=mask_dropout,
                            same_latent=same_latent,
                            perform_injection=inject,
                            cache_cpu_offloading=True,cf_clip=info.get("cf_clip"))
                        samples_list.append(extra_image)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
                            
            elif infer_mode == "kolor_face":
                samples_list = []
                for key in role_list:
                   for emb in only_role_emb[key]: 
                        samples = model(
                            prompt=None,
                            prompt_embeds=emb[0],
                            negative_prompt_embeds=negative[0],
                            pooled_prompt_embeds=emb[1],
                            negative_pooled_prompt_embeds=negative[1],
                            height=height,
                            width=width,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            num_images_per_prompt=1,
                            generator=torch.Generator(device=device).manual_seed(seed),
                            face_crop_image=input_id_img_s_dict[key][0],
                            face_insightface_embeds=input_id_emb_s_dict[key][0].to(device, dtype=torch.float16),
                        ).images
                        #print(samples.shape)
                        samples_list.append(samples)
                if nc_emb:
                    for emb in nc_emb:
                        samples = model(
                            prompt=None,
                            prompt_embeds=emb[0],
                            negative_prompt_embeds=negative[0],
                            pooled_prompt_embeds=emb[1],
                            negative_pooled_prompt_embeds=negative[1],
                            height=height,
                            width=width,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            num_images_per_prompt=1,
                            generator=torch.Generator(device=device).manual_seed(seed),
                            face_crop_image=Image.new('RGB', (336, 336), (255, 255, 255)),
                            face_insightface_embeds=torch.zeros_like(input_id_emb_s_dict[role_list[0]][0]).to(device, dtype=torch.float16),
                        ).images
                        #print(samples.shape)
                        samples_list.append(samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
           
            elif infer_mode == "infiniteyou": 
                samples_list = []

                for key in role_list:
                    if isinstance(input_id_img_s_dict[key][0], list) and (len(input_id_img_s_dict[key][0]) < len(only_role_emb[key])):#输入的图片数量小于emb数量
                        cn_img=input_id_img_s_dict[key][0]+(len(only_role_emb[key])-len(input_id_img_s_dict[key][0]))*[control_image_zero]
                    else:
                        cn_img=input_id_img_s_dict[key][0]
                  
                    for index,emb in enumerate(only_role_emb[key]):       
                        
                        samples = model (id_embed=input_id_emb_s_dict[key][0],
                            prompt_embeds=emb[0],
                            pooled_prompt_embeds=emb[1].get("pooled_output"),
                            control_image=cn_img[index] if isinstance(cn_img , list) else input_id_img_s_dict[key][0],
                            guidance_scale=cfg,
                            num_steps=steps,
                            seed=seed,
                            infusenet_conditioning_scale=1.0,
                            infusenet_guidance_start=0,
                            infusenet_guidance_end=1.0,
                            height=height,
                            width=width,
                            )
                        samples_list.append(samples)  
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
            elif infer_mode == "flux_pulid":
                samples_list = []
                for key in role_list:
                    for index,emb in enumerate(only_role_emb[key]):
                        samples = model.generate_image(width=width, 
                            height=height,
                            num_steps=steps,
                            start_step=2,
                            guidance=cfg,
                            seed=seed,
                            inp=emb,
                            inp_neg=negative[0][key][index],
                            x=negative[1][key][index], #seed 上一个节点调用
                            id_embeddings=input_id_emb_s_dict[key][0],
                            uncond_id_embeddings=input_id_emb_un_dict[key][0],
                            )  # torch.Size([1, 4, 64, 64])
                    
                        samples_list.append(samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
            elif  infer_mode=="uno":
                samples_list = []
                for key in role_list:
                    for index,emb in enumerate(only_role_emb[key]):
                        #print(emb)
                        samples = model(width=width, 
                            height=height,guidance=cfg,
                            num_steps=steps,
                            inp_cond=emb,
                            )  # torch.Size([1, 4, 64, 64])
                    
                        samples_list.append(samples)

                if daul_emb:
                    for index, emb in zip(dual_index, daul_emb):
                        samples = model(width=width, 
                            height=height,guidance=cfg,
                            num_steps=steps,
                            inp_cond=emb,
                            )  # torch.Size([1, 4, 64, 64])
                        samples_list.insert(index, samples)

                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
            else: #none:
                return

class StoryDiffusion_Lora_Control:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL"),
                             "switch": ("DIFFCONDI"),
                             "loras": (["none"] + folder_paths.get_filename_list("loras"),),
                             "trigger_words": ("STRING", {"default": "best quality"}),
                             "controlnets": (["none"] + folder_paths.get_filename_list("controlnet"),),
                             "controlnet_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                             },    
                }
    
    RETURN_TYPES = ("MODEL","DIFFCONDI",)
    RETURN_NAMES = ("model","switch",)
    FUNCTION = "main_apply"
    CATEGORY = "Storydiffusion"
    
    def main_apply(self,model,switch,loras,trigger_words,controlnets,controlnet_scale):
        use_lora,controlnet,lora_path =False, None,None
        infer_mode=switch.get("infer_mode")

        if infer_mode=="story_maker" or infer_mode=="msdiffusion" or infer_mode=="story_and_maker":
            if loras != "none":
                lora_path = folder_paths.get_full_path("loras", loras)
                active_lora = model.get_active_adapters() 
                if active_lora:
                    print(f'active_lora is :{active_lora}')
                    model.unload_lora_weights()  # make sure lora is not mix
                if os.path.basename(lora_path) in lora_lightning_list:
                    model.load_lora_weights(lora_path)
                else:
                    model.load_lora_weights(lora_path, adapter_name=trigger_words)
                use_lora=True
            if controlnets != "none":
                controlnet_path = folder_paths.get_full_path("controlnet", controlnets)
                from diffusers import ControlNetModel
                from safetensors.torch import load_file
                controlnet = ControlNetModel.from_unet(model.unet)
                cn_state_dict = load_file(controlnet_path, device="cpu")
                controlnet.load_state_dict(cn_state_dict, strict=False)
                del cn_state_dict
                gc_cleanup()
                controlnet.to(torch.float16)
                if infer_mode=="story_maker" :
                    model.controlnet = controlnet
        elif infer_mode=="infiniteyou" :
            if loras != "none":
                lora_path = folder_paths.get_full_path("loras", loras)
            if lora_path is not None:
                use_lora=True
                loras = []
                if "realism" in lora_path:
                    loras.append([lora_path, 'realism', 1.0])# single only now
                if  "blur" in lora_path:
                    loras.append([lora_path, 'anti_blur', 1.0])
                model.load_loras(loras)
        switch["controlnet_scale"]=controlnet_scale
        switch["controlnet"]=controlnet
        switch["use_lora"]=use_lora
        switch["trigger_words"]=trigger_words

        return (model,switch)



NODE_CLASS_MAPPINGS = {
    "Pre_Translate_prompt": Pre_Translate_prompt,
    "Comic_Type": Comic_Type,
    "EasyFunction_Lite":EasyFunction_Lite,
    "StoryDiffusion_Apply":StoryDiffusion_Apply,
    "StoryDiffusion_CLIPTextEncode":StoryDiffusion_CLIPTextEncode,
    "StoryDiffusion_KSampler":StoryDiffusion_KSampler,
    "StoryDiffusion_Lora_Control":StoryDiffusion_Lora_Control,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pre_Translate_prompt": "Pre_Translate_prompt",
    "Comic_Type": "Comic_Type",
    "EasyFunction_Lite":"EasyFunction_Lite",
    "StoryDiffusion_Apply":"StoryDiffusion_Apply",
    "StoryDiffusion_CLIPTextEncode":"StoryDiffusion_CLIPTextEncode",
    "StoryDiffusion_KSampler":"StoryDiffusion_KSampler",
    "StoryDiffusion_Lora_Control":"StoryDiffusion_Lora_Control"
}
