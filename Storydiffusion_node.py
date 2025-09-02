 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import logging
import numpy as np
import torch
import os
from PIL import ImageFont,Image
import torch.nn.functional as F
import copy
from pathlib import PureWindowsPath
from tqdm import tqdm
from .utils.utils import get_comic
from .model_loader_utils import  (phi2narry,replicate_data_by_indices,get_float,gc_cleanup,tensor_to_image,photomaker_clip,tensortopil_list_upscale,tensortopil_list,extract_content_from_brackets_,
                                  narry_list_pil,pre_text2infer,cf_clip,get_phrases_idx_cf,get_eot_idx_cf,get_ms_phrase_emb,get_extra_function,photomaker_clip_v2,adjust_indices,load_clip_clipvsion,
                                  get_scheduler,apply_style_positive,load_lora_for_unet_only,tensortolist,
                                  nomarl_upscale,SAMPLER_NAMES,SCHEDULER_NAMES,lora_lightning_list)

from .utils.gradio_utils import cal_attn_indice_xl_effcient_memory,is_torch2_available
from .ip_adapter.attention_processor import IPAttnProcessor2_0
if is_torch2_available():
    from .utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from .utils.gradio_utils import AttnProcessor

import folder_paths
from comfy.model_management import total_vram
import comfy
import latent_preview


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))


weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) # use gguf dir

global total_count, attn_count_, cur_step, mask1024, mask4096, attn_procs_, unet_,sa32, sa64,write,height_s, width_s

infer_type_g=torch.float16 if device=="cuda" else torch.float32 #TODO

 

class EasyFunction_Lite:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "repo1": ("STRING", { "default": ""}),
                            "repo2": ("STRING", { "default": ""}),
                            "unet":(["none"] +folder_paths.get_filename_list("diffusion_models"),),
                            "gguf": (["none"] + folder_paths.get_filename_list("gguf"),),
                            "clip1": (["none"] + folder_paths.get_filename_list("clip"),),
                            "clip2": (["none"] + folder_paths.get_filename_list("clip"),),
                            "clip_vision1": (["none"] + folder_paths.get_filename_list("clip_vision"),),
                            "clip_vision2": (["none"] + folder_paths.get_filename_list("clip_vision"),),
                            "lora1": (["none"] +folder_paths.get_filename_list("loras"),),
                            "lora2": (["none"] +folder_paths.get_filename_list("loras"),),
                            "controlnet":(["none"] +folder_paths.get_filename_list("controlnet"),),
                            "special_mode": (["none","tag", "glm"],),
                            "tag_temperature": (
                                 "FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                             }}
    
    RETURN_TYPES = ("MODEL","CLIP","STORY_CONDITIONING_1")
    RETURN_NAMES = ("model","clip","info")
    FUNCTION = "easy_function_main"
    CATEGORY = "Storydiffusion"
    
    def easy_function_main(self, repo1,repo2,unet,gguf,clip1,clip2,clip_vision1,clip_vision2,lora1,lora2,controlnet,special_mode,tag_temperature):
       
        repo1_path=PureWindowsPath(repo1).as_posix() if repo1 else None
        repo2_path=PureWindowsPath(repo2).as_posix() if repo2 else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        unet_path=folder_paths.get_full_path("unet", unet) if unet != "none" else None
        clip1_path=folder_paths.get_full_path("clip", clip1) if clip1 != "none" else None
        clip2_path=folder_paths.get_full_path("clip", clip2) if clip2 != "none" else None
        lora1_path  =folder_paths.get_full_path("loras", lora1) if lora1 != "none" else None
        lora2_path  =folder_paths.get_full_path("loras", lora2) if lora2 != "none" else None
        clip_vision1_path=folder_paths.get_full_path("clip_vision", clip_vision1) if clip_vision1 != "none" else None
        clip_vision2_path=folder_paths.get_full_path("clip_vision", clip_vision2) if clip_vision2 != "none" else None
        controlnet_path=folder_paths.get_full_path("controlnet", controlnet) if controlnet != "none" else None
 
        clip_glm,tag_model,pipe,svdq_repo=None,None,None,None

        repo_list=[i for i in [repo1_path,repo2_path] if i is not None]

        if repo1_path is not None or repo2_path is not None:

            find_svdq=[i for i in repo_list if "svdq" in i ]
            svdq_repo=find_svdq[0] if find_svdq else None
   

        if special_mode=="tag":
            from .model_loader_utils import StoryLiteTag 
            tag_model = StoryLiteTag(device, tag_temperature,repo2_path, repo1_path) #No repo will load default
                    
        elif special_mode=="glm": #kolor only
            from .model_loader_utils import GLM_clip
            if clip1_path is not None:
                clip_glm=GLM_clip(dir_path,clip1_path)
            elif clip2_path is not None and clip1_path is None:
                clip_glm=GLM_clip(dir_path,clip2_path)
            else:
                clip_glm=None
        else:
            pass


        info={"gguf_path":gguf_path,"unet_path":unet_path,"tag_model":tag_model,"clip_vision1_path":clip_vision1_path,"controlnet_path":controlnet_path,
              "clip1_path":clip1_path,"clip2_path":clip2_path,"repo1_path":repo1_path,"repo2_path":repo2_path,"svdq_repo":svdq_repo,
              "lora1_path":lora1_path,"lora2_path":lora2_path,"clip_vision2_path":clip_vision2_path,}
        
        return (pipe,clip_glm,info)


class StoryDiffusion_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "infer_mode": (["story", "classic","flux_pulid","infiniteyou","uno","realcustom","instant_character","dreamo","qwen_image","flux_omi","bagel_edit","story_maker","story_and_maker","consistory","kolor_face","msdiffusion" ],),
                    "photomake_ckpt": (["none"] + [i for i in folder_paths.get_filename_list("photomaker") if "v1" in i or "v2" in i],),
                    "ipadapter_ckpt": (["none"] + folder_paths.get_filename_list("photomaker"),),
                    "quantize_mode": ([ "fp8", "nf4","fp16", ],),
                    "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                    "extra_function":("STRING", {"default": ""}),
                            },
                "optional":{
                   
                    "info": ("STORY_CONDITIONING_1",),
                    "CLIP_VISION": ("CLIP_VISION",),
                            }
                }
    
    RETURN_TYPES = ("MODEL","DIFFCONDI",)
    RETURN_NAMES = ("model","switch",)
    FUNCTION = "main_apply"
    CATEGORY = "Storydiffusion"
    
    def main_apply(self,model,vae,infer_mode,photomake_ckpt,ipadapter_ckpt,quantize_mode,lora_scale,extra_function, **kwargs):
        print(f"infer model is {infer_mode}")
        extra_info=kwargs.get("info",{})
       
        # pre data
        photomake_ckpt_path = None if photomake_ckpt == "none"  else  folder_paths.get_full_path("photomaker", photomake_ckpt)
        ipadapter_ckpt_path = None if ipadapter_ckpt == "none"  else  folder_paths.get_full_path("photomaker", ipadapter_ckpt)

        if extra_function:
            extra_function=PureWindowsPath(extra_function).as_posix()

        save_quantezed=True if "save" in extra_function and quantize_mode=="fp8" else False

        clip_vision1_path=extra_info.get("clip_vision1_path") if extra_info else None
        clip_vision2_path=extra_info.get("clip_vision2_path") if extra_info else None
        repo1_path=extra_info.get("repo1_path",None) 
        repo2_path=extra_info.get("repo2_path",None) 
        lora1_path=extra_info.get("lora1_path") if extra_info else None
        lora2_path=extra_info.get("lora2_path") if extra_info else None
        unet_path=extra_info.get("unet_path") if extra_info else None
        gguf_path=extra_info.get("gguf_path") if extra_info else None

        repo_list=[i for i  in [repo1_path,repo2_path] if i is not None]
        lora_list=[i for i in [lora1_path,lora2_path] if i is not None]
        
        dreamo_version="v1.0" if "v1.0" in extra_function else "v1.1"

        vae_encoder,vae_downsample_factor,vae_config,vision_model_config_ar,image_proj_model,no_dif_quantization,find_Kolors=None,None,None,None,None,False,None
    
        # per clip vision
        CLIP_VISION=kwargs.get("CLIP_VISION") 
        unet_type=torch.float16 #use for sdxl
        
        if infer_mode=="flux_pulid" or infer_mode=="kolor_face":# 2种加载clip vision的方式 
            from comfy.clip_vision import load as clip_load
            if CLIP_VISION is not None:
                clip_vision_path=CLIP_VISION
            elif  clip_vision1_path is not None:    
                clip_vision_path=clip_load(clip_vision1_path).model
            elif clip_vision2_path is not None:  
                clip_vision_path=clip_load(clip_vision2_path).model
            else:
                if infer_mode=="kolor_face":
                    pass
                else:
                    raise ValueError("Please specify one of CLIP_VISION or clip_vision1_path or clip_vision2_path")

        if infer_mode=="msdiffusion" or infer_mode in ["story_maker" ,"story_and_maker"]:
            if CLIP_VISION is not None:
                pass
            else:
                from comfy.clip_vision import load as clip_load
                if  clip_vision1_path is not None:    
                    CLIP_VISION=clip_load(clip_vision1_path).model
                elif clip_vision2_path is not None:  
                    CLIP_VISION=clip_load(clip_vision2_path).model
                else:
                    raise ValueError("Please provide a CLIP_VISION or CLIP_VISION1 or CLIP_VISION2,Msdiffusion need a clipvison g model,story_maker need a clipvison H model")
                
        # pre dreamo lora
        if infer_mode =="dreamo" and  lora1_path is not None and lora2_path is not None:
            if "distill" in lora1_path.lower():
                cfg_distill_path=lora1_path
                dreamo_lora_path=lora2_path
            else:
                cfg_distill_path=lora2_path
                dreamo_lora_path=lora1_path
        else:
            cfg_distill_path=None
            dreamo_lora_path=None

        
        if infer_mode in ["story_maker" ,"story_and_maker"] and ipadapter_ckpt_path is None:
             raise "story_maker need a mask.bin"

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
        edit_mode=None
        logging.info(f"total_vram is {total_vram},aggressive_offload is {aggressive_offload},offload is {offload}")

        if infer_mode in["story", "story_and_maker","msdiffusion"]:# mix mode,use maker or ms to make 2 roles in on image
            from .model_loader_utils import Loader_storydiffusion
            model = Loader_storydiffusion(model,photomake_ckpt_path,vae)
        elif infer_mode =="story_maker":
            from .model_loader_utils import Loader_story_maker
            model = Loader_story_maker(model,ipadapter_ckpt_path,vae,False,lora_scale)
        elif infer_mode == "flux_pulid":
            from .PuLID.app_flux import get_models
            from .model_loader_utils import Loader_Flux_Pulid
            if unet_path is None:
                raise "PuLID can't link a normal comfyui model,you need load a flux unet model "
            model_=get_models("flux-dev",unet_path,False,aggressive_offload,device=device,offload=offload,quantized_mode=quantize_mode,)
            model = Loader_Flux_Pulid(model_,model,ipadapter_ckpt_path,quantize_mode,aggressive_offload,offload,False,clip_vision_path)
        elif infer_mode == "infiniteyou":
            from .model_loader_utils import Loader_InfiniteYou
            assert extra_info ,"you need to provide extra_info"
            model,image_proj_model = Loader_InfiniteYou(extra_info,vae,quantize_mode)
        elif infer_mode == "consistory":
            from .model_loader_utils import load_pipeline_consistory
            model = load_pipeline_consistory(model,vae)
        elif infer_mode == "instant_character":
            from .model_loader_utils import load_pipeline_instant_character
            assert extra_info ,"you need to provide extra_info"
            model = load_pipeline_instant_character(extra_info,ipadapter_ckpt_path,vae,quantize_mode)
        elif infer_mode == "realcustom":
            from .model_loader_utils import load_pipeline_realcustom,load_realcustom_vae
            if ipadapter_ckpt_path is None:
                raise "realcustom need a realcustom model which in photomaker folder, and  chocie it in ipadapter_ckpt_path"
            model,vision_model_config_ar,_ = load_pipeline_realcustom(model,ipadapter_ckpt_path)
            vae_encoder,vae_downsample_factor,vae_config=load_realcustom_vae(vae,device)
        elif infer_mode == "kolor_face":
            from .model_loader_utils import Loader_KOLOR
           
            find_Kolors =[i for i in repo_list if  "kolor"  in i.lower()]
            if not find_Kolors:
                raise ValueError("No Kolor model found in the repo")
            model = Loader_KOLOR(find_Kolors[0],clip_vision_path,ipadapter_ckpt_path) 
        elif infer_mode == "uno":
            from .model_loader_utils import Loader_UNO
            model = Loader_UNO(extra_info,offload,quantize_mode,save_quantezed,lora_rank=512)
        elif infer_mode == "dreamo":
            from.model_loader_utils import Loader_Dreamo
            if dreamo_lora_path is None or cfg_distill_path is None or ipadapter_ckpt_path is None:
                raise "dreamo need a dreamo lora and cfg distill and turbo lora in ipadapter menu"
            model = Loader_Dreamo(extra_info,vae,quantize_mode,dreamo_lora_path,cfg_distill_path,ipadapter_ckpt_path,device,dreamo_version)
        elif infer_mode == "bagel_edit":
            from .Bagel.app import load_bagel_model
          
            if not repo_list :
                raise "EasyFunction_Lite node repo1 or repo2 must fill bagel repo"
            max_mem_per_gpu=str(int(total_vram/1000))+"GIB"
            model = load_bagel_model(repo_list[0],quantize_mode,max_mem_per_gpu)
        elif infer_mode == "flux_omi":
            from .model_loader_utils import Loader_Flux_Diffuser
        
            no_dif_quantization=True if extra_info.get("unet_path") or extra_info.get("svdq_repo") or extra_info.get("gguf_path") else False
            model = Loader_Flux_Diffuser(extra_info,ipadapter_ckpt_path,vae,quantize_mode)
        elif infer_mode == "qwen_image":
            from.qwen_image.inferencer import load_quwen_image   
            
            df_repo=repo_list[0] if repo_list else None
            
            model,edit_mode = load_quwen_image(cpu_offload=True,cpu_offload_blocks=16,no_pin_memory=True, dir_path =dir_path, repo=df_repo,unet_path=unet_path,gguf_path=gguf_path,lora_path=lora_list[0] if lora_list else None) 
        else:  # can not choice a mode
            print("infer use comfyui classic mode")
        
        story_img=True if photomake_ckpt_path and infer_mode in["story","story_maker","story_and_maker","msdiffusion"] else False
        model_=model if infer_mode=="flux_pulid" or story_img else None
        switch={"infer_mode":infer_mode,"ipadapter_ckpt_path":ipadapter_ckpt_path,"photomake_ckpt_path":photomake_ckpt_path,
                "vision_model_config_ar":vision_model_config_ar,"no_dif_quantization":no_dif_quantization,"edit_mode":edit_mode,
                       "lora_scale":lora_scale,"image_proj_model":image_proj_model, "vae_encoder":vae_encoder,"vae_downsample_factor":vae_downsample_factor,"vae_config":vae_config,"dreamo_version":dreamo_version,
                       "CLIP_VISION":CLIP_VISION,"VAE":vae,"find_Kolors":find_Kolors,"model_":model_,"unet_type":unet_type,"extra_function":extra_function,}
        switch.update(extra_info)
        return (model,switch,)

      
class StoryDiffusion_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "switch": ("DIFFCONDI", ),
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
            "optional": {
                         "image":("IMAGE",),
                         "control_image":("IMAGE",),
                         }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","DIFFINFO","INT","INT",)
    RETURN_NAMES = ("positive", "negative","condition","width","height",)
    FUNCTION = "encode"
    CATEGORY = "Storydiffusion"


    def encode(self, clip,switch,width,height, role_text,scene_text,pos_text,neg_text,lora_trigger_words,add_style,mask_threshold,extra_param,guidance_list,**kwargs):
        infer_mode=switch.get("infer_mode")
        CLIP_VISION = switch.get("CLIP_VISION")
        extra_function=switch.get("extra_function")
        photomake_ckpt_path=switch.get("photomake_ckpt_path")
        unet_type=switch.get("unet_type")
        model_=switch.get("model_")
        vae=switch.get("VAE")


        image=kwargs.get("image",None)
        control_image=kwargs.get("control_image",None)

        lora_list=[i for i in [switch.get("lora1_path"),switch.get("lora2_path")] if i is not None]
        repo_list=[i for i in [switch.get("repo1_path"),switch.get("repo2_path")] if i is not None]

        use_lora =True if lora_list else False


        if extra_function:
            extra_function=PureWindowsPath(extra_function).as_posix()
        tag_list,text_model,vision_model,siglip_path,dino_path=None,None,None,None,None,

        # 反推功能
        if switch.get("tag_model") is not None and isinstance(image,torch.Tensor):
            tag_img_list=tensortopil_list(image)
            tag_list=[]
            for i in tag_img_list:
                tag_text=switch.get("tag_model").run_tag(i)
                tag_list.append(tag_text)




        siglip_path_=switch.get("clip_vision1_path") 
        dino_path_=switch.get("clip_vision2_path")
        clip1_path=switch.get("clip1_path")
        clip2_path=switch.get("clip2_path")


        siglip_path, dino_path = siglip_path_, dino_path_

        if (siglip_path_ is not None and dino_path_ is not None and 
            "sig" in dino_path_ and "dino" in siglip_path_):
            siglip_path, dino_path = dino_path_, siglip_path_i
            

        if infer_mode == "realcustom" and (siglip_path is None or dino_path is None):
            print ("if use realcustom mode, u must linke add_function to your node,if not ,will auto load clip_vision_path and clip_path")
            siglip_path="" if siglip_path is None else siglip_path
            dino_path="" if dino_path is None else dino_path
                
    
        auraface,use_photov2,img2img_mode,cached,inject,onnx_provider,dreamo_mode,trigger_words_dual,dual_lora_scale=get_extra_function(extra_function,extra_param,photomake_ckpt_path,image,infer_mode)
        

        
        (replace_prompts,role_index_dict,invert_role_index_dict,ref_role_index_dict,ref_role_totals,role_list,role_dict,
         nc_txt_list,nc_indexs,positions_index_char_1,positions_index_char_2,positions_index_dual,prompts_dual,index_char_1_list,index_char_2_list)=pre_text2infer(role_text,scene_text,lora_trigger_words,use_lora,tag_list)


        global character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals, character_dict
        character_index_dict=role_index_dict
        invert_character_index_dict=invert_role_index_dict
        ref_indexs_dict=ref_role_index_dict
        ref_totals=ref_role_totals
        character_dict=role_dict
        
        _, style_neg = apply_style_positive(add_style, " ") #get n
        neg_text = neg_text + style_neg
        
        replace_prompts=[i+pos_text for i in replace_prompts]
        
        only_role_list=[apply_style_positive(add_style,i)[0] for i in replace_prompts]
        

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
        
        input_id_images_dict={}
        image_list=[]
        
        if img2img_mode and image is not None :
             if len(role_list)==1:
                image_pil=tensortopil_list_upscale(image, width, height)
                input_id_images_dict[role_list[0]]=image_pil[0]
                image_list=[image_pil[0]]
             else:
                f1, _, _, _ = image.size()
                img_list = list(torch.chunk(image, chunks=f1))
                image_list = [nomarl_upscale(img, width, height) for img in img_list]
                for index, key in enumerate(role_list):
                    input_id_images_dict[key]=image_list[index]   

        if img2img_mode:
            if infer_mode == "story_and_maker" or infer_mode == "story_maker":
                k1, _, _, _ = image.size()
                if k1 == 1:
                    image_emb = [CLIP_VISION.encode_image(image)["penultimate_hidden_states"]]
                else:
                    img_list = list(torch.chunk(image, chunks=k1))
                    image_emb=[]
                    for i in img_list:
                        image_emb.append(CLIP_VISION.encode_image(i)["penultimate_hidden_states"].to(device, dtype=unet_type))   
            elif infer_mode=="story":
                image_emb = None  # story的图片模式的emb要从ip的模型里单独拿，
            elif infer_mode=="msdiffusion":
                image_emb = CLIP_VISION.encode_image(image)["penultimate_hidden_states"] #MS分情况，图生图直接在前面拿，文生图在在sample拿
                if not image_emb.is_cuda:#确保emb在cuda,以及type的正确
                    image_emb = image_emb.to(device,dtype=unet_type)
            elif infer_mode == "instant_character":
                from .model_loader_utils import load_dual_clip,instant_character_id_clip
                if extra_function is not None and extra_param:
                    if "sig" in extra_function and  "dino" in extra_param: #TODO 
                        siglip_path_i=extra_function
                        dino_path_i=extra_param
                    else:
                        siglip_path_i=extra_param
                        dino_path_i=extra_function
                else:
                    dino_path_i="facebook/dinov2-giant"
                    siglip_path_i="google/siglip-so400m-patch14-384"
                siglip_image_encoder,siglip_image_processor,dino_image_encoder_2,dino_image_processor_2=load_dual_clip(siglip_path_i,dino_path_i,device,torch.bfloat16)
                image_emb=[]
                for img in tensortopil_list(image):
                    id_emb=instant_character_id_clip(img,siglip_image_encoder,siglip_image_processor,dino_image_encoder_2,dino_image_processor_2,device,torch.bfloat16)
                    image_emb.append(id_emb)
                siglip_image_encoder.to("cpu")
                dino_image_encoder_2.to("cpu")
                gc_cleanup()
            elif infer_mode == "flux_pulid":

                # get emb use insightface
                pass
            elif infer_mode == "bagel_edit":
                image_emb=input_id_images_dict
            elif infer_mode == "flux_omi":
                image_emb=input_id_images_dict
            elif infer_mode == "dreamo":
                from .model_loader_utils import Dreamo_image_encoder
                from huggingface_hub import hf_hub_download

                BEN2_path_list=[i for i in repo_list if 'BEN2_Base.pth' in i]
                if not BEN2_path_list:
                    BEN2_path= hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='ComfyUI/models')
                else:
                    BEN2_path=BEN2_path_list[0]
                ref_list=tensortopil_list_upscale(image,width,height)
                if control_image is not None and dreamo_mode=="id":#如果是id加ip模式，角色可以多次换装，但是要考虑衣服的数量
                    control_pil_list=tensortopil_list_upscale(control_image,width,height)
                    if len(control_pil_list) != len(only_role_list):
                        raise "when use dreamo id + ip,control image must same size as prompts,dreamo的id模式,多少句角色prompt,就要有多少件衣服在control节点输入"
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
                else:
                    control_pil_list=None

                #task_list=["ip", "id", "style"] #TODO,
                image_emb={}
                if control_pil_list is not None and dreamo_mode=="id":
                    for key,role_img ,control_list in zip(role_list,ref_list,control_pil_list):
                        id_role_emb=[]
                        for i in control_list:
                            role_emb=Dreamo_image_encoder(BEN2_path,role_img,i,dreamo_mode,"ip",ref_res=512) #TODO
                            id_role_emb.append(role_emb)
                        image_emb[key]=id_role_emb
                else:
                    for key,role_img in zip(role_list,ref_list):
                        role_emb=Dreamo_image_encoder(BEN2_path,role_img,None,dreamo_mode,"ip",ref_res=512) #TODO
                        image_emb[key]=[role_emb]*len(only_role_list) #改成列表方便协同id模式

            elif infer_mode == "realcustom":
                if "g" in os.path.basename(clip1_path).lower():
                    clip1_path, clip2_path = clip2_path, clip1_path

                text_model,vision_model=load_clip_clipvsion([clip1_path,clip2_path],
                                                            [os.path.join(dir_path, "config/clip_1"),os.path.join(dir_path, "config/clip_2")],
                                                            dino_path,siglip_path,switch.get("vision_model_config_ar"))
                
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
  

        # pre insight face model and emb
        if img2img_mode:
            if infer_mode in ["story_and_maker","story_maker","flux_pulid","kolor_face","infiniteyou"] or (use_photov2 and infer_mode=="story"):

                from .model_loader_utils import insight_face_loader,get_insight_dict
                
                
                find_mask=[i for i in repo_list if "rmgb" in i.lower()] if repo_list else None
                if find_mask is not None:
                    mask_repo=find_mask[0]
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

          

        # pre clip txt emb
      
        noise_x,inp_neg_list,letent_real=[],[],{}
        
        if  infer_mode=="consistory":
            only_role_emb=None
        elif infer_mode=="instant_character":
            from .model_loader_utils import cf_flux_prompt_clip
            only_role_emb={}
            for key ,prompts in zip(role_list,inf_list_split):
                emb_list_=[]
                for prompt in prompts:
                    p_,pool_,ind_=cf_flux_prompt_clip(clip,prompt)
                    emb_list_.append([p_,pool_,ind_])
                only_role_emb[key]=emb_list_

        elif infer_mode=="realcustom":
            from .model_loader_utils import realcustom_clip_emb
            samples_per_prompt=1
            guidance_weight=3.5
            role_text_list = role_text.splitlines()
            roel_text_c=''.join(role_text_list) 
            
            if '(' in roel_text_c and ')' in roel_text_c:
                    object_prompt = extract_content_from_brackets_(roel_text_c)  # 提取prompt的object list
                    #print(f"object_prompt:{object_prompt}")
                    object_prompt=[i.strip() for i in object_prompt]
                    for i in object_prompt:
                        if " " in i:
                            raise "when using [object],object must be a word,any blank in it will cause error."
                        
                    object_prompt=[i for i in object_prompt ]
                    target_phrases = sorted(list(set(object_prompt)),key=lambda x: list(object_prompt).index(x))  # 清除同名物体,保持原有顺序
                    #print(f"object_prompt:{phrases}",len(phrases))
                    assert  len(target_phrases)>=2,"when using msdiffusion ,object must be more than 2."
                    if len(target_phrases)>2:
                        target_phrases=target_phrases[:2] #只取前两个物体
            else:
                raise "when using realcustom ,(objectA)  and (objectA) must be in the role prompt."
            
            print(f"object_prompt:{target_phrases}")
            image_list=tensortopil_list_upscale(image, width, height)

            only_role_emb,letent_real={},{}
          
            for key ,prompts,role_image,target_phrase in zip(role_list,inf_list_split,image_list,target_phrases):
                emb_dict_real_list,latent_dict_list=[],[]
                for p,n in zip(prompts,[neg_text]*len(prompts)): 
                  
                    emb_dict_real,latent_dict=realcustom_clip_emb(text_model,vision_model,switch.get("vae_config"),switch.get("vae_downsample_factor"),p,n,role_image,target_phrase,
                                                                width,height,device,samples_per_prompt,guidance_weight)
                    emb_dict_real_list.append(emb_dict_real)
                    latent_dict_list.append(latent_dict)
                only_role_emb[key]=emb_dict_real_list
                letent_real[key]=latent_dict_list
            vision_model.to("cpu")
            gc_cleanup()         
        elif infer_mode=="flux_pulid":
            from .PuLID.flux.util import load_clip, load_t5
            from .PuLID.app_flux  import get_emb_flux_pulid

            #repo_in="flux-dev" if not repo else repo
            if_repo =False
            t5_ = load_t5("flux-dev",clip,if_repo,device, max_length=128)
            clip_ = load_clip("flux-dev",clip,if_repo,device)
            only_role_emb,noise_x,inp_neg_list={},{},{}
            
            for key ,prompts in zip(role_list,inf_list_split):
                ip_emb,inp_n=[],[]
                
                for p,n in zip(prompts,[neg_text]*len(prompts)): 
                    inp,inp_neg=get_emb_flux_pulid(t5_,clip_,if_repo,p,n,width,height,num_steps=20,guidance=3.5,device=device)
                    ip_emb.append(inp)
                    inp_n.append(inp_neg)
                only_role_emb[key]=ip_emb
                inp_neg_list[key]=inp_n
        elif infer_mode == "uno":
            only_role_emb={}
            from .UNO.uno.flux.sampling import prepare_multi_ip_wrapper
            
            for key ,prompts in zip(role_list,inf_list_split):
                ip_emb=[]
                for p,x_1 in zip(prompts,x_1_refs_dict[key]):
                    inp = prepare_multi_ip_wrapper(clip,prompt=p, ref_imgs=x_1, pe=uno_pe,device=device,h=height,w=width)
                    ip_emb.append(inp)
                only_role_emb[key]=ip_emb
        elif infer_mode=="kolor_face":
            from .model_loader_utils import glm_single_encode
            from .kolors.models.tokenization_chatglm import ChatGLMTokenizer
            
            tokenizer = ChatGLMTokenizer.from_pretrained(os.path.join(switch.get("find_Kolors"),'text_encoder'))
            assert clip is not None, "clip is None,check your clip path"
            chatglm3_model = {
                'text_encoder': clip, 
                'tokenizer': tokenizer
                }
            only_role_emb,only_role_emb_ne=glm_single_encode(chatglm3_model, inf_list_split,role_list, neg_text, 1) 
        elif infer_mode=="flux_omi" and  switch.get("no_dif_quantization"): #need comfyclip
            from .model_loader_utils import cf_flux_prompt_clip
            only_role_emb={}
            for key ,prompts in zip(role_list,inf_list_split):
                emb_list_=[]
                for prompt in prompts:
                    p_,pool_,ind_=cf_flux_prompt_clip(clip,prompt)
                    emb_list_.append([p_,pool_,ind_])
                only_role_emb[key]=emb_list_

        else:
            if photomake_ckpt_path is not None and img2img_mode and infer_mode in["story","story_and_maker","msdiffusion"]: #img2img模式下SDXL的story的clip要特殊处理，有2个imgencoder进程，所以分离出来 TODO
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
                elif infer_mode=="dreamo" or infer_mode=="bagel_edit" or (infer_mode=="flux_omi" and not switch.get("no_dif_quantization")):
                    pass # TODO 暂时不支持dreamo
                elif infer_mode=="qwen_image":
                    from .qwen_image.inferencer import get_emb_data
                    image_list_=tensortolist(image,width,height)
                    only_role_emb= get_emb_data(clip,vae,inf_list_split,image_list_,role_list)
                else:
                    only_role_emb= cf_clip(inf_list_split, clip, infer_mode,role_list)  #story,story_maker,story_and_maker,msdiffusion,infinite
                    if len (role_list)==1 :
                        only_role_emb_dict={}
                        only_role_emb_dict[role_list[0]]=only_role_emb # [ [cond_p, output_p],...]
                        only_role_emb=only_role_emb_dict
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
                elif infer_mode=="dreamo" or infer_mode=="bagel_edit" or infer_mode=="flux_omi":
                    pass # TODO 暂时不支持dreamo

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
                if '(' in prompts_dual[0] and ')' in prompts_dual[0]:
                    object_prompt = extract_content_from_brackets_(prompts_dual[0])  # 提取prompt的object list
                    #print(f"object_prompt:{object_prompt}")
                    object_prompt=[i.strip() for i in object_prompt]
                    for i in object_prompt:
                        if " " in i:
                            raise "when using [object],object must be a word,any blank in it will cause error."
                        
                    object_prompt=[i for i in object_prompt ]
                    phrases = sorted(list(set(object_prompt)),key=lambda x: list(object_prompt).index(x))  # 清除同名物体,保持原有顺序
                    #print(f"object_prompt:{phrases}",len(phrases))
                    assert  len(phrases)>=2,"when using msdiffusion ,object must be more than 2."
                    if len(phrases)>2:
                        phrases=phrases[:2] #只取前两个物体
                else:
                    raise "when using msdiffusion ,(objectA)  and (objectA) must be in the prompt."
                if use_lora:
                    prompts_dual=[i+lora_trigger_words for i in prompts_dual]
                
                prompts_dual=[apply_style_positive(add_style,i+pos_text)[0] for i in prompts_dual] #[' T a (pig)  play whith  a (doll) in the garden,best 8k,RAW']

                
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
                inp = prepare_multi_ip_wrapper(clip,prompt=dual_t, ref_imgs=x_1, pe=uno_pe,device=device,h=height,w=width)
                daul_emb.append(inp)
        elif prompts_dual and infer_mode == "dreamo":
            from .model_loader_utils import Dreamo_image_encoder
            from huggingface_hub import hf_hub_download
            BEN2_path_list=[i for i in repo_list if "BEN2_Base.pth" in i]
            if not BEN2_path_list:
                BEN2_path= hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='ComfyUI/models')
            else:
                BEN2_path=BEN2_path_list[0]
            ref_list=tensortopil_list_upscale(image,width,height)
            #task_list=["ip", "id", "style"] #TODO
            images_emb=Dreamo_image_encoder(BEN2_path,ref_list[0],ref_list[1],"ip","ip",ref_res=512) #TODO
            prompts_dual=[i.replace(role_list[0] ,role_dict[role_list[0]]) for i in prompts_dual if role_list[0] in i ]
            prompts_dual = [i.replace(role_list[1], role_dict[role_list[1]]) for i in prompts_dual if role_list[1] in i]
            prompts_dual=[apply_style_positive(add_style,i+pos_text)[0] for i in prompts_dual] #[' The figurine  play whith  The pig in the garden,best 8k,RAW']
            daul_emb=[images_emb,prompts_dual]
        else:
            daul_emb=None
        # neg
        if infer_mode=="consistory":
            negative = None
            postive_dict={}
        elif infer_mode=="instant_character":
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb} 
            negative = None
        elif infer_mode=="dreamo":
            only_role_emb={}
            for key ,prompts in zip(role_list,inf_list_split):
                only_role_emb[key]=prompts
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb}
            negative = neg_text # TODO
        elif infer_mode=="bagel_edit":
            only_role_emb={}
            for key ,prompts in zip(role_list,inf_list_split):
                only_role_emb[key]=prompts
            postive_dict= {"role": only_role_emb, "nc": None, "daul": None}
            negative = neg_text # TODO
        elif infer_mode=="flux_omi":
            if switch.get("no_dif_quantization"):
                postive_dict= {"role": only_role_emb, "nc": None, "daul": None} #only_role_emb:p_,pool_,ind_
                negative = neg_text # TODO
            else:
                only_role_emb={}
                for key,prompts in zip(role_list,inf_list_split):
                    only_role_emb[key]=prompts
                postive_dict= {"role": only_role_emb, "nc": None, "daul": None}
                negative = neg_text # TODO
        elif infer_mode=="realcustom":
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb} 
            negative=[letent_real]
        elif infer_mode=="flux_pulid":
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb} #不支持NC
            negative = [inp_neg_list,noise_x]
        elif infer_mode=="uno":
            postive_dict= {"role": only_role_emb, "nc": None, "daul": daul_emb} #TODO
            negative=None
        elif infer_mode=="kolor_face":
            postive_dict = {"role": only_role_emb, "nc": nc_emb, "daul": daul_emb}
            negative = only_role_emb_ne[0]
        elif infer_mode=="qwen_image":
            postive_dict = {"role": only_role_emb, "nc": nc_emb, "daul": daul_emb}
            from .qwen_image.inferencer import get_emb_data

            image_list_=tensortolist(image, width, height) if not switch.get("edit_mode") else None
            neg_list=[[neg_text]] if len(role_list)==1 else [[neg_text],[neg_text]]
            negative= get_emb_data(clip,vae,neg_list,image_list_,role_list)

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
        
            
        # Pre emb for maker
        
        if img2img_mode and infer_mode in ["story_and_maker","story_maker"]:
            from .StoryMaker.pipeline_sdxl_storymaker_wrapper import encode_prompt_image_emb_
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
                                                                                             cloth_2,device, num_images_per_prompt, unet_type, CLIP_VISION,vae,do_classifier_free_guidance=True)
                            p_list.append(prompt_image_emb_)
                            maker_cn_list.append(maker_control_image_)
                        prompt_image_emb[key]=p_list#输出改为字典
                        maker_control_image[key]=maker_cn_list#输出改为字典
                    # 还要将每个元素转为列表，改为[img,cn_img]

                else:
                    prompt_image_emb,maker_control_image={},{}
                    
                    for i,key in enumerate(role_list):
                        prompt_image_emb_,maker_control_image_=encode_prompt_image_emb_(make_img[i], image_2, make_mask_img[i], mask_image_2, make_face_info[i], face_info_2, make_cloth_info[i], cloth_2,device, num_images_per_prompt, unet_type, CLIP_VISION,vae,do_classifier_free_guidance=True)
                        prompt_image_emb[key]=[prompt_image_emb_]#输出改为字典
                        maker_control_image[key]=[maker_control_image_]#输出改为字典
      
            else:
                prompt_image_emb_1,maker_control_image_1=encode_prompt_image_emb_(make_img[0], image_2, make_mask_img[0], mask_image_2, make_face_info[0], face_info_2, make_cloth_info[0], cloth_2,device, num_images_per_prompt, unet_type, CLIP_VISION,vae,do_classifier_free_guidance=True)
                
                prompt_image_emb={role_list[0]:len(only_role_list)*[prompt_image_emb_1]}#输出改为字典
                maker_control_image={role_list[0]:len(only_role_list)*[maker_control_image_1]}#输出改为字典

            maker_control_image_dual=None
            if daul_emb is not None:
                prompt_image_emb_dual,maker_control_image_dual=encode_prompt_image_emb_(make_img[0], make_img[1], make_mask_img[0], make_mask_img[1], make_face_info[0], make_face_info[1], make_cloth_info[0], make_cloth_info[1],device, num_images_per_prompt, unet_type, CLIP_VISION,vae,do_classifier_free_guidance=True)
                prompt_image_emb_dual=len(prompts_dual)*[prompt_image_emb_dual]
                maker_control_image_dual=len(prompts_dual)*[maker_control_image_dual] 
        else: 
            
            prompt_image_emb,maker_control_image,prompt_image_emb_dual,maker_control_image_dual=None,None,None,None

              
        
        # switch
        new_dict={
            "id_len":len(role_list),"role_list":role_list,"invert_role_index_dict":invert_role_index_dict,"image_list":image_list,"inf_list_split":inf_list_split,
            "nc_index":nc_indexs,"dual_index":positions_index_dual,"grounding_kwargs":grounding_kwargs,"cross_attention_kwargs":cross_attention_kwargs,
            "image_embeds":image_emb,"img2img_mode":img2img_mode,"positions_index_char_1":positions_index_char_1,"dreamo_mode":dreamo_mode,
            "positions_index_char_2":positions_index_char_2,"mask_threshold":mask_threshold,"lora_list":lora_list,
            "input_id_emb_s_dict":input_id_emb_s_dict,"input_id_img_s_dict":input_id_img_s_dict,"input_id_emb_un_dict":input_id_emb_un_dict,"trigger_words_dual":trigger_words_dual,
            "maker_control_image":maker_control_image,"prompt_image_emb":prompt_image_emb,"prompt_image_emb_dual":prompt_image_emb_dual,"dual_lora_scale":dual_lora_scale,
            "maker_control_image_dual":maker_control_image_dual,"prompts_dual":prompts_dual,"control_image":control_image,"trigger_words":lora_trigger_words,
            "only_role_list":only_role_list,"nc_txt_list":nc_txt_list,"neg_text":neg_text,"role_text":role_text,"cf_clip":clip,"cached":cached,
            "inject":inject,"role_key_list":role_key_list,"input_id_images_dict":input_id_images_dict,"id_index":(index_char_1_list,index_char_2_list)
        }
        switch.update(new_dict)
    
        
 
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
                "condition":("DIFFINFO",{
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
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,condition, latent_image,sa32_degree,sa64_degree, denoise=1.0, **kwargs):
       
        infer_mode=condition.get("infer_mode")
        id_len=condition.get("id_len")
        nc_index=condition.get("nc_index")
        dual_index=condition.get("dual_index")
        role_list=condition.get("role_list")
        mask_threshold=condition.get("mask_threshold")
        grounding_kwargs=condition.get("grounding_kwargs")
        cross_attention_kwargs=condition.get("cross_attention_kwargs")
        image_embeds=condition.get("image_embeds")
        img2img_mode=condition.get("img2img_mode")
        num_images_per_prompt=1
        photomake_ckpt_path = condition.get("photomake_ckpt_path")
        ipadapter_ckpt_path = condition.get("ipadapter_ckpt_path")
        prompt_image_emb=condition.get("prompt_image_emb")
        maker_control_image=condition.get("maker_control_image")
        prompt_image_emb_dual=condition.get("prompt_image_emb_dual")
        maker_control_image_dual=condition.get("maker_control_image_dual")
        controlnet=condition.get("controlnet")
        prompts_dual=condition.get("prompts_dual")
        input_id_emb_un_dict=condition.get("input_id_emb_un_dict")
        input_id_img_s_dict=condition.get("input_id_img_s_dict")
        input_id_emb_s_dict=condition.get("input_id_emb_s_dict")
        input_id_cloth_dict=condition.get("input_id_cloth_dict")
        only_role_list=condition.get("only_role_list")
        nc_txt_list=condition.get("nc_txt_list")
        neg_text=condition.get("neg_text")
        cached=condition.get("cached")
        inject=condition.get("inject")
        inf_list_split=condition.get("inf_list_split")
        input_id_images_dict=condition.get("input_id_images_dict")
        dreamo_mode=condition.get("dreamo_mode")
        invert_role_index_dict=condition.get("invert_role_index_dict")
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

            lora_list=condition.get("lora_list",[])
            trigger_words=condition.get("trigger_words","best")     
            trigger_words_dual=condition.get("trigger_words_dual","best")
            lora_scale=condition.get("lora_scale",1.0)
            dual_lora_scale=condition.get("dual_lora_scale",1.0)
            if infer_mode in ["story_maker" ,"msdiffusion" ,"story_and_maker" ,"story",]:
                
                if lora_list:
                    if len(lora_list)==2:
                        lora_list_l = [ i for i  in lora_list if  os.path.basename(i) in lora_lightning_list]

                        if lora_list_l:
                            if 1==len(lora_list_l):
                                model=load_lora_for_unet_only(model,lora_list_l[0],trigger_words)
                              
                            else:
                                for i,j,k in zip(lora_list_l,[trigger_words,trigger_words_dual],[lora_scale,dual_lora_scale]):
                                    model=load_lora_for_unet_only(model,i,j,k)
                        else:
                            for i,j ,k in zip(lora_list,[trigger_words,trigger_words_dual],[lora_scale,dual_lora_scale]):
                                 model=load_lora_for_unet_only(model,i,j,k)
                    else:
                        model=load_lora_for_unet_only(model,lora_list[0],trigger_words,lora_scale)

                if condition.get("controlnet_path")  is not None and infer_mode!="story":
                    controlnet_path =condition.get("controlnet") 
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



            #get emb
            only_role_emb=positive.get("role")
            nc_emb=positive.get("nc")
            daul_emb=positive.get("daul")
            daul_emb_ms=daul_emb
            
            if infer_mode in["story" ,"msdiffusion","story_and_maker"]: #三者都调用story的unet方法，只是双角色引入ms或者maker
                if ipadapter_ckpt_path is None and infer_mode=="msdiffusion":
                    raise "msdiffusion  need a ms_adapter.bin file at ipadapter_ckpt menu."
                global attn_procs_,sa32, sa64, write, height_s, width_s,attn_count_, total_count, id_length, total_length, cur_step,cur_character

                sa32 = sa32_degree
                sa64 = sa64_degree
                attn_count_ = 0
                total_count = 0
                cur_step = 0
                id_length = id_len
                total_length = 5
                attn_procs_ = {}
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
                       
                        VAE=condition.get("VAE")
                        for j ,(index, i) in enumerate(zip(dual_index, daul_emb_ms)):
                            seed_random = random.randint(0, seed)
                            write = False
                            if not img2img_mode: #文生图模式，以文生图第一张为ID参考拿emb
                                CLIP_VISION = condition.get("CLIP_VISION")
                                
                                out_1, out_2 = {}, {}
                                out_1["samples"]=samples_list[int(condition.get("positions_index_char_1"))]
                                out_2["samples"] =samples_list[int(condition.get("positions_index_char_2"))]
                                role_1=VAE.decode(out_1["samples"])
                                role_2 = VAE.decode(out_2["samples"])
                               
                                if  infer_mode=="msdiffusion":
                                    from .model_loader_utils import Infer_MSdiffusion
                                    role_tensor=torch.cat((role_1,role_2),dim=0)
                                    image_embeds=CLIP_VISION.encode_image(role_tensor)["penultimate_hidden_states"]
                                    image_embeds = image_embeds.to(device, dtype=model.unet.dtype)
                                    if controlnet:
                                        model=model #TO DO
                                    #write = False
                                    samples = Infer_MSdiffusion(model,ipadapter_ckpt_path,image_embeds, i[0],negative[0],grounding_kwargs,cross_attention_kwargs,
                                                                 1,mask_threshold,height,width,steps,seed_random,cfg,i[1].get("pooled_output"),negative[1].get("pooled_output") )

                                else: #maker的emb要单独处理，用列表形式传入 make
                                    from .StoryMaker.pipeline_sdxl_storymaker_wrapper import encode_prompt_image_emb_
                                    # image_embeds=[]
                                    # for X in [role_1,role_2]:
                                    #     image_embeds.append(clip_vision.encode_image(X)["penultimate_hidden_states"].to(device, dtype=model.unet.dtype))
                                    make_img,make_mask_img,make_face_info,make_cloth_info=[],[],[],[]
                                    from .model_loader_utils import insight_face_loader,get_insight_dict
                                    app_face,pipeline_mask,app_face_=insight_face_loader(infer_mode,False, False)
                                   
                                    image_list = [tensor_to_image(role_1),tensor_to_image(role_2)]
                                    input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,input_id_cloth_dict=get_insight_dict(app_face,app_face_,pipeline_mask,infer_mode,False,image_list,role_list,condition.get("control_image"),width, height) 
                                    for key in role_list:
                                        img_ = input_id_emb_un_dict[key][0]
                                        # print(character_key_str,input_id_images_dict)
                                        mask_image_ = input_id_img_s_dict[key][0] #mask_image
                                        face_info_ = input_id_emb_s_dict[key][0]
                                        cloth_info_ = None
                                        if isinstance(condition.get("control_image"), torch.Tensor):
                                            cloth_info_ = input_id_cloth_dict[key][0]
                                        make_img.append(img_)
                                        make_mask_img.append(mask_image_)
                                        make_face_info.append(face_info_)
                                        make_cloth_info.append(cloth_info_)

                                    prompt_image_emb_dual,maker_control_image_dual=encode_prompt_image_emb_(make_img[0], make_img[1], make_mask_img[0], make_mask_img[1], make_face_info[0],
                                                                                                             make_face_info[1], make_cloth_info[0], make_cloth_info[1],device, num_images_per_prompt, condition.get("unet_type"), CLIP_VISION,VAE,do_classifier_free_guidance=True)
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
                                    from .model_loader_utils import Loader_storydiffusion,Infer_MSdiffusion
                                    model=Loader_storydiffusion(None,None,None,model)
                                    gc_cleanup()
                                    samples = Infer_MSdiffusion(model,ipadapter_ckpt_path,image_embeds, i[0],negative[0],grounding_kwargs,cross_attention_kwargs,
                                                                 1,mask_threshold,height,width,steps,seed_random,cfg,i[1].get("pooled_output"),negative[1].get("pooled_output") )
                                #del model
                                else:
                                    from .model_loader_utils import Loader_story_maker
                                    print("reload maker") 
                                    model=Loader_story_maker(None,ipadapter_ckpt_path,VAE,False,condition.get("lora_scale"),UNET=model)
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
                role_input=condition.get("role_text").splitlines()[0]
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
                                                         same_latent=same_latent, perform_injection=inject,n_achors=n_achors,cf_clip=condition.get("cf_clip"))
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
                        cache_cpu_offloading=True,cf_clip=condition.get("cf_clip"))
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
                            cache_cpu_offloading=True,cf_clip=condition.get("cf_clip"))
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
                            seed=random.randint(0, seed),
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
                        samples = model.generate_image(
                            width=width, 
                            height=height,
                            num_steps=steps,
                            start_step=2,
                            guidance=cfg,
                            seed=random.randint(0, seed),
                            inp=emb,
                            inp_neg=negative[0][key][index],
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
                        samples = model(
                            width=width, 
                            height=height,
                            guidance=cfg,
                            num_steps=steps,
                            inp_cond=emb,
                            seed=random.randint(0, seed),
                            
                            )  # torch.Size([1, 4, 64, 64])
                    
                        samples_list.append(samples)

                if daul_emb:
                    for index, emb in zip(dual_index, daul_emb):
                        samples = model(
                            width=width, 
                            height=height,
                            guidance=cfg,
                            num_steps=steps,
                            inp_cond=emb,
                            seed=random.randint(0, seed) 
                            )  # torch.Size([1, 4, 64, 64])
                        samples_list.insert(index, samples)

                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)
                return (out,)
            elif infer_mode=="realcustom": 
                from .model_loader_utils import realcustom_infer
                write = False
                samples_list = []
                for key in role_list:
                    for emb,latent_ in zip(only_role_emb[key],negative[0][key]):
                        samples = realcustom_infer(
                            model,
                            sample_steps=steps, 
                            mask_reused_step=12,
                            emb_dict=emb,
                            latent_dict=latent_,
                            mask_scope=0.20,
                            mask_strategy="max_norm", #"min_max_per_channel", "max_norm"
                            guidance_weight=cfg,
                            height=height,
                            width=width,
                            seed=random.randint(0, seed),
                            device=device,
                            )
                        samples_list.append(samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)  
                return (out,) 
            elif infer_mode=="instant_character": 
                samples_list = []
                #model.to(device)
                for key,id_emb in zip(role_list,image_embeds):
                    for emb_list in only_role_emb[key]:
                        samples = model(
                            prompt_embeds=emb_list[0], 
                            pooled_prompt_embeds=emb_list[1],
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            subject_image=True,
                            subject_scale=0.9,
                            generator=torch.manual_seed(random.randint(0, seed)),
                            text_ids=emb_list[2],
                            subject_image_embeds_dict=id_emb,
                            )[0]  # torch.Size([1, 4, 64, 64])
                        #print(samples.shape)
                        samples_list.append(samples)
        
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)  
                return (out,) 
            elif infer_mode=="dreamo": 
                samples_list = []
                cfg=4.5 if condition.get("dreamo_version")=="v1.1" else 3.5
        
                first_step_guidance=0
                for key in role_list: 
                    for index,prompt in tqdm(enumerate(only_role_emb[key]),desc=f"Processing {key}"):
                        samples = model(prompt=prompt,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            ref_conds=image_embeds[key][index],
                            generator=torch.Generator(device="cpu").manual_seed(seed),
                            true_cfg_scale=1,
                            true_cfg_start_step=0,
                            true_cfg_end_step=0,
                            negative_prompt=neg_text,
                            neg_guidance_scale=cfg,
                            first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else cfg,
                        ).images
                        #print(samples.shape)
                        samples_list.append(samples)
                if daul_emb:
                    for index, prompt in tqdm(zip(dual_index, daul_emb[1]),desc="Processing dual prompts"): #daul_emb [emb,prompts]
                        samples = model(prompt=prompt,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            ref_conds=daul_emb[0],
                            generator=torch.Generator(device="cpu").manual_seed(seed),
                            true_cfg_scale=1,
                            true_cfg_start_step=0,
                            true_cfg_end_step=0,
                            negative_prompt=neg_text,
                            neg_guidance_scale=cfg,
                            first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else cfg,
                        ).images
                        samples_list.insert(index, samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)  
                return (out,) 
            elif infer_mode =="bagel_edit":
                
                VAE=condition.get("VAE")
                samples_list = []
                if img2img_mode:
                    from .Bagel.app import edit_image
                    for key in role_list: 
                        for prompt in only_role_emb[key]:
                            samples,text_=edit_image(
                                model,
                                image=image_embeds[key],
                                prompt=prompt,
                                show_thinking=False, 
                                cfg_text_scale=cfg, 
                                cfg_img_scale=2.0,
                                cfg_interval=0.0, 
                                timestep_shift=3.0,
                                num_timesteps=steps,
                                cfg_renorm_min=0.0, 
                                cfg_renorm_type="text_channel",
                                max_think_token_n=1024, 
                                do_sample=False,
                                text_temperature=0.3,
                                seed=seed) # tuple (image,str)
                            print(f'thinking text is {text_}')
                            samples_list.append(samples)
                else:
                    from .Bagel.app import text_to_image
                    if width==height: #todo
                        image_ratio="1:1"
                    elif width/height==0.75:
                        image_ratio="4:3"
                    elif width/height==3/4:
                        image_ratio="3:4"
                    elif width/height==9/16:
                        image_ratio="9:16"
                    elif width/height==16/9:
                        image_ratio="16:9"
                    else:
                        image_ratio="1:1"

                    for key in role_list: 
                        for prompt in only_role_emb[key]:
                            samples,text_=text_to_image(
                                model,
                                prompt=prompt,
                                show_thinking=False, 
                                cfg_text_scale=cfg, 
                                cfg_interval=0.4,
                                timestep_shift=3.0,
                                num_timesteps=steps,
                                cfg_renorm_min=0.0,
                                cfg_renorm_type="global",
                                max_think_token_n=1024,
                                do_sample=False,
                                text_temperature=0.3,
                                seed=seed,
                                image_ratio=image_ratio) # tuple (image,str)
                            print(f'thinking text is {text_}')
                            samples_list.append(samples)
                            
                out = {}
                samples_list=[VAE.encode(phi2narry(pixels)[:,:,:,:3]) for pixels in samples_list] # TODO pil to tensor
                out["samples"] = torch.cat(samples_list, dim=0)  
                return (out,) 
            elif infer_mode =="flux_omi":
                samples_list = []
                if condition.get("no_dif_quantization"):
                    for key in role_list: 
                        for ebm_list in only_role_emb[key]:
                            samples=model(
                                prompt=None,
                                height=height,
                                width=width, 
                                guidance_scale=cfg if cfg==3.5 else 3.5, 
                                num_inference_steps=steps,
                                max_sequence_length=512,
                                generator=torch.Generator("cpu").manual_seed(seed),
                                prompt_embeds=ebm_list[0],
                                pooled_prompt_embeds=ebm_list[1],
                                spatial_images=[image_embeds[key]],
                                subject_images=[], #TODO TRY ON
                                cond_size=512,
                                ).images
                            #print(samples.shape)
                            samples_list.append(samples)
                else:
                    for key in role_list: 
                        for prompt in only_role_emb[key]:
                            samples=model(
                                prompt=prompt,
                                height=height,
                                width=width, 
                                guidance_scale=cfg if cfg==3.5 else 3.5, 
                                num_inference_steps=steps,
                                max_sequence_length=512,
                                generator=torch.Generator("cpu").manual_seed(seed),
                                spatial_images=[image_embeds[key]],
                                subject_images=[], #TODO TRY ON
                                cond_size=512,
                                ).images
                            #print(samples.shape)
                            samples_list.append(samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)  
                # Clear cache after generation
                from .OmniConsistency.infer import clear_cache
                clear_cache(model.transformer)
                return (out,)
            elif infer_mode =="qwen_image":
                from .qwen_image.inferencer import infer_qwen_image_edit
                samples_list = []
                # import comfy.model_management as model_management
                # model_management.unload_all_models()
             
                image_list=condition.get("image_list") if  condition.get("edit_mode") else [None,None]
                for key,image in zip(role_list,image_list): 
                    negative_prompt_embeds,_=negative[key][0]
                   
                    for prompt in only_role_emb[key]:
                        samples=infer_qwen_image_edit(
                            model,
                            image=image,
                            prompt_embeds=prompt[0],
                            prompt_embeds_mask=torch.zeros_like(prompt[0]),
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_prompt_embeds_mask=torch.zeros_like(negative_prompt_embeds),
                            seed=seed,
                            num_inference_steps=steps,
                            true_cfg_scale=cfg,
                            image_latents=prompt[1],
                            edit_mode=condition.get("edit_mode"),
                            latents=latent_init,
                            )
                        #print(samples.shape)
                        samples_list.append(samples)
                out = {}
                out["samples"] = torch.cat(samples_list, dim=0)  
                return (out,)
            else: #none:
                return


    
class Comic_Type:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "scene_prompts": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                             "fonts_list": (os.listdir(os.path.join(dir_path, "fonts")),),
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

   
def set_attention_processor(unet_, id_length, is_ipadapter=False):
    global attn_procs_
    attn_procs_ = {}
    for name in unet_.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet_.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet_.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet_.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet_.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs_[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs_[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs_[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet_.device, dtype=torch.float16)
            else:
                attn_procs_[name] = AttnProcessor()

    unet_.set_attn_processor(copy.deepcopy(attn_procs_))

    
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
        global total_count, attn_count_, cur_step, indices1024, indices4096
        global sa32, sa64
        global write
        global height_s, width_s
        global character_dict
        global  character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals
        if attn_count_ == 0 and cur_step == 0:
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
        attn_count_ += 1
        if attn_count_ == total_count:
            attn_count_ = 0
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


NODE_CLASS_MAPPINGS = {
    "Pre_Translate_prompt": Pre_Translate_prompt,
    "Comic_Type": Comic_Type,
    "EasyFunction_Lite":EasyFunction_Lite,
    "StoryDiffusion_Apply":StoryDiffusion_Apply,
    "StoryDiffusion_CLIPTextEncode":StoryDiffusion_CLIPTextEncode,
    "StoryDiffusion_KSampler":StoryDiffusion_KSampler,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pre_Translate_prompt": "Pre_Translate_prompt",
    "Comic_Type": "Comic_Type",
    "EasyFunction_Lite":"EasyFunction_Lite",
    "StoryDiffusion_Apply":"StoryDiffusion_Apply",
    "StoryDiffusion_CLIPTextEncode":"StoryDiffusion_CLIPTextEncode",
    "StoryDiffusion_KSampler":"StoryDiffusion_KSampler",
}

