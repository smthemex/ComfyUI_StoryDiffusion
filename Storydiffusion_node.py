# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import gc
import logging
import numpy as np
import torch
import os

from PIL import ImageFont
from huggingface_hub import hf_hub_download
from diffusers import (StableDiffusionXLPipeline, DiffusionPipeline,EulerDiscreteScheduler, UNet2DConditionModel,UniPCMultistepScheduler, AutoencoderKL,)
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

import folder_paths
from comfy.model_management import cleanup_models
from comfy.clip_vision import load as clip_load
from comfy.model_management import total_vram

from .utils.utils import get_comic
from .utils.gradio_utils import character_to_dict
from .utils.load_models_utils import load_models, get_instance_path
from .model_loader_utils import  (story_maker_loader,kolor_loader,phi2narry,
                                  extract_content_from_brackets,narry_list,remove_punctuation_from_strings,phi_list,center_crop_s,center_crop,
                                  narry_list_pil,setup_seed,find_directories,
                                  apply_style,get_scheduler,set_attention_processor,load_character_files_on_running,
                                  save_results,nomarl_upscale,process_generation,SAMPLER_NAMES,SCHEDULER_NAMES,lora_lightning_list,pre_checkpoint,get_easy_function)

global total_count, attn_count, cur_step, mask1024, mask4096, attn_procs, unet
global sa32, sa64
global write
global height_s, width_s

photomaker_dir=os.path.join(folder_paths.models_dir, "photomaker")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))

fonts_path = os.path.join(dir_path, "fonts")
fonts_lists = os.listdir(fonts_path)

base_pt = os.path.join(photomaker_dir,"pt")
if not os.path.exists(base_pt):
    os.makedirs(base_pt)
    

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
                         "control_image": ("IMAGE",),
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
        print(f"Process using {id_number} roles....")
        image = kwargs.get("image")
        
        if isinstance(image,torch.Tensor):
            print(image.shape)
            batch_num,_,_,_=image.size()
            model_type="img2img"
            if batch_num!=id_number:
                raise "role prompt numbers don't match input image numbers...example:2 roles need 2 input images,"
        else:
            model_type = "txt2img"
            image=None
       
        if controlnet_model=="none":
            controlnet_path=None
            control_image=None
        else:
            controlnet_path=folder_paths.get_full_path("controlnet", controlnet_model)
            control_image=kwargs.get("control_image")
            if not isinstance(control_image, torch.Tensor):
                raise "if using controlnet,need input a image in control_image"
        
        photomaker_path = os.path.join(photomaker_dir, f"photomaker-{photomake_mode}.bin")
        photomake_mode_=photomake_mode
       
        # load model
        (auraface, NF4, save_model, kolor_face,flux_pulid_name,pulid,quantized_mode,story_maker,make_dual_only,
         clip_vision_path,char_files,ckpt_path,lora,lora_path,use_kolor,photomake_mode,use_flux,onnx_provider)=get_easy_function(
            easy_function,clip_vision,character_weights,ckpt_name,lora,repo_id,photomake_mode)
        
        
        photomaker_path,face_ckpt,photomake_mode,pulid_ckpt,face_adapter,kolor_ip_path=pre_checkpoint(
            photomaker_path, photomake_mode, kolor_face, pulid, story_maker, clip_vision_path,use_kolor,model_type)
    
        if total_vram > 30000.0:
            aggressive_offload = False
            offload = False
        elif 18000.0 < total_vram < 30000.0:
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
                raise "Now,using comfyUI normal processing must need comfyUI clip,if using flux need dual clip ."
            use_cf = True
            use_flux=False
            if cf_model.model.model_type.value==8:
                use_flux = True
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
            
        elif not repo_id and ckpt_path: # load ckpt
            if_repo = False
            if story_maker:
                if not make_dual_only: #default dual
                    logging.info("start story-make processing...")
                    pipe=story_maker_loader(clip_load,clip_vision_path,dir_path,ckpt_path, face_adapter,UniPCMultistepScheduler)
                else:
                    photomake_mode_1 =  photomake_mode_
                    logging.info("start story-diffusion and story-make processing...")
                    pipe = load_models(ckpt_path, model_type=model_type, single_files=True, use_safetensors=True,
                                       photomake_mode=photomake_mode_1, photomaker_path=photomaker_path, lora=lora,
                                       lora_path=lora_path,
                                       trigger_words=trigger_words, lora_scale=lora_scale)
                    set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            elif "flux" in ckpt_path.lower():
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
                                         vae_cf=vae_path,if_repo=if_repo,onnx_provider=onnx_provider)
                else:
                    raise "need pulid in easy function"
            else:
                logging.info("start story-diffusion  processing...")
                pipe = load_models(ckpt_path, model_type=model_type, single_files=True, use_safetensors=True,
                                   photomake_mode=photomake_mode, photomaker_path=photomaker_path, lora=lora,
                                   lora_path=lora_path,
                                   trigger_words=trigger_words, lora_scale=lora_scale)
                set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
        else: #if repo or  no ckpt,choice repo
            if_repo=True
            if repo_id.rsplit("/")[-1].lower()=="playground-v2.5-1024px-aesthetic":
                logging.info("start playground story-diffusion  processing...")
                pipe = DiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float16,
                )
                set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            elif repo_id.rsplit("/")[-1].lower()=="sdxl-unstable-diffusers-y":
                logging.info("start sdxl-unstable story-diffusion  processing...")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    repo_id, torch_dtype=torch.float16,use_safetensors=False
                )
                set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
            elif use_kolor:
                logging.info("start kolor processing...")
                pipe=kolor_loader(repo_id, model_type, set_attention_processor, id_length, kolor_face, clip_vision_path,
                             clip_load, CLIPVisionModelWithProjection, CLIPImageProcessor,
                             photomaker_dir, face_ckpt, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel)
                pipe.enable_model_cpu_offload()
            elif use_flux:
                from .model_loader_utils import flux_loader
                pipe=flux_loader(folder_paths,ckpt_path,repo_id,AutoencoderKL,save_model,model_type,pulid,clip_vision_path,NF4,vae_id,offload,aggressive_offload,pulid_ckpt,quantized_mode,
                if_repo,dir_path,clip,onnx_provider)
                if lora:
                    if not "Hyper" in lora_path : #can't support Hyper now
                        if not NF4:
                            logging.info("try using lora in flux quantize processing...")
                            pipe.load_lora_weights(lora_path)
                            pipe.fuse_lora(lora_scale=0.125)  # lora_scale=0.125
                       
            else: # SD dif_repo
                if  story_maker:
                    if not make_dual_only:
                        logging.info("start story_maker processing...")
                        from .StoryMaker.pipeline_sdxl_storymaker import StableDiffusionXLStoryMakerPipeline
                        pipe = StableDiffusionXLStoryMakerPipeline.from_pretrained(
                            repo_id, torch_dtype=torch.float16)
                        if device != "mps":
                            pipe.cuda()
                        image_encoder=clip_load(clip_vision_path)
                        pipe.load_storymaker_adapter(image_encoder, face_adapter, scale=0.8, lora_scale=0.8)
                        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                else:
                    logging.info("start story_diffusion processing...")
                    pipe = load_models(repo_id, model_type=model_type, single_files=False, use_safetensors=True,
                                       photomake_mode=photomake_mode,
                                       photomaker_path=photomaker_path, lora=lora,
                                       lora_path=lora_path,
                                       trigger_words=trigger_words, lora_scale=lora_scale)
                    set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
                    
        if vae_id != "none":
            if not use_flux and not use_kolor and not use_cf:
                vae_id = folder_paths.get_full_path("vae", vae_id)
                vae_config=os.path.join(dir_path, "local_repo","vae")
                pipe.vae=AutoencoderKL.from_single_file(vae_id, config=vae_config,torch_dtype=torch.float16)
        load_chars = False
        if not use_kolor and not use_flux and not use_cf:
            if story_maker:
                if make_dual_only:
                    pipe.scheduler = scheduler_choice.from_config(pipe.scheduler.config)
                    load_chars = load_character_files_on_running(pipe.unet, character_files=char_files)
            else:
                pipe.scheduler = scheduler_choice.from_config(pipe.scheduler.config)
                load_chars = load_character_files_on_running(pipe.unet, character_files=char_files)
            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            pipe.enable_vae_slicing()
            pipe.to(device)
        # if device != "mps":
        #     pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        # need get emb
        character_name_dict_, character_list_ = character_to_dict(character_prompt, lora, trigger_words)
        #print(character_list_)
        miX_mode = False
        if model_type=="img2img":
            d1, _, _, _ = image.size()
            if d1 == 1:
                image_load = [nomarl_upscale(image, width, height)]
            else:
                img_list = list(torch.chunk(image, chunks=d1))
                image_load = [nomarl_upscale(img, width, height) for img in img_list]
                
            from .model_loader_utils import insight_face_loader,get_insight_dict
            app_face,pipeline_mask,app_face_=insight_face_loader(photomake_mode, auraface, kolor_face, story_maker, make_dual_only, photomake_mode_)
            input_id_emb_s_dict, input_id_img_s_dict, input_id_emb_un_dict, input_id_cloth_dict=get_insight_dict(app_face,pipeline_mask,app_face_,image_load,photomake_mode,
                                                                                                                 kolor_face,story_maker,make_dual_only,photomake_mode_,
                     pulid,pipe,character_list_,control_image,width, height)
        else:
            input_id_emb_s_dict = {}
            input_id_img_s_dict = {}
            input_id_emb_un_dict = {}
            input_id_cloth_dict = {}
        #print(input_id_img_s_dict)
        role_name_list = [i for i in character_name_dict_.keys()]
        #print( role_name_list)
        model={"pipe":pipe,"use_flux":use_flux,"use_kolor":use_kolor,"photomake_mode":photomake_mode,"trigger_words":trigger_words,"lora_scale":lora_scale,
               "load_chars":load_chars,"repo_id":repo_id,"lora_path":lora_path,"ckpt_path":ckpt_path,"model_type":model_type, "lora": lora,
               "scheduler":scheduler,"width":width,"height":height,"kolor_face":kolor_face,"pulid":pulid,"story_maker":story_maker,
               "make_dual_only":make_dual_only,"face_adapter":face_adapter,"clip_vision_path":clip_vision_path,
               "controlnet_path":controlnet_path,"character_prompt":character_prompt,"image":image,"control_image":control_image,
               "input_id_emb_s_dict":input_id_emb_s_dict,"input_id_img_s_dict":input_id_img_s_dict,"use_cf":use_cf,
               "input_id_emb_un_dict":input_id_emb_un_dict,"input_id_cloth_dict":input_id_cloth_dict,"role_name_list":role_name_list,"miX_mode":miX_mode}
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
            }

      
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "prompt_array",)
    FUNCTION = "story_sampler"
    CATEGORY = "Storydiffusion"

    def story_sampler(self, model,scene_prompts, negative_prompt, img_style, seed, steps,
                  cfg, denoise_or_ip_sacle, style_strength_ratio,
                  guidance, mask_threshold, start_step,save_character,controlnet_scale,guidance_list,):
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
        control_image=model.get("control_image")
        image=model.get("image")
        use_cf=model.get("use_cf")
        input_id_emb_s_dict = model.get("input_id_emb_s_dict")
        input_id_img_s_dict = model.get("input_id_img_s_dict")
        input_id_emb_un_dict = model.get("input_id_emb_un_dict")
        input_id_cloth_dict = model.get("input_id_cloth_dict")
        role_name_list=model.get("role_name_list")
        miX_mode=model.get("miX_mode")
        
        cf_scheduler=scheduler
        #print(input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,role_name_list) #'[Taylor]',['[Taylor]']
    
        empty_emb_zero = None
        if model_type=="img2img":           
            if pulid or kolor_face or photomake_mode=="v2":
                empty_emb_zero = torch.zeros_like(input_id_emb_s_dict[role_name_list[0]][0]).to(device)
        
        # 格式化文字内容
        scene_prompts.strip()
        character_prompt.strip()
        # 从角色列表获取角色方括号信息
        char_origin = character_prompt.splitlines()
        char_origin=[i for i in char_origin if "[" in i]
        #print(char_origin)
        char_describe = char_origin # [A a men...,B a girl ]
        char_origin = [char.split("]")[0] + "]" for char in char_origin]
        #print(char_origin)
        # 判断是否有双角色prompt，如果有，获取双角色列表及对应的位置列表，
        prompts_origin = scene_prompts.splitlines()
        prompts_origin=[i.strip() for i in prompts_origin]
        prompts_origin = [i for i in prompts_origin if "[" in i]
        #print(prompts_origin)
        positions_dual = [index for index, prompt in enumerate(prompts_origin) if len(extract_content_from_brackets(prompt))>=2]  #改成单句中双方括号方法，利于MS组句，[A]... [B]...[C]
        prompts_dual = [prompt for prompt in prompts_origin if len(extract_content_from_brackets(prompt))>=2]
        
        if len(char_origin) == 2:
            positions_char_1 = [index for index, prompt in enumerate(prompts_origin) if char_origin[0] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置
            positions_char_2 = [index for index, prompt in enumerate(prompts_origin) if char_origin[1] in prompt][
                0]  # 获取角色出现的索引列表，并获取首次出现的位置
            
        if model_type=="img2img":
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
                                     kolor_face,pulid,story_maker,input_id_emb_s_dict, input_id_img_s_dict,input_id_emb_un_dict, input_id_cloth_dict,guidance,control_image,empty_emb_zero,miX_mode,use_cf,cf_scheduler)

        else:
            if story_maker:
                raise "story maker only suppport img2img now"
            upload_images = None
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
                                     trigger_words,photomake_mode,use_kolor,use_flux,make_dual_only,kolor_face,
                                     pulid,story_maker,input_id_emb_s_dict, input_id_img_s_dict,input_id_emb_un_dict, input_id_cloth_dict,guidance,control_image,empty_emb_zero,miX_mode,use_cf,cf_scheduler)

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
                    cleanup_models(keep_clone_weights_loaded=False)
                    gc.collect()
                    torch.cuda.empty_cache()
                    pipe=story_maker_loader(clip_load,clip_vision_path,dir_path,ckpt_path,face_adapter,UniPCMultistepScheduler)
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
                if isinstance(control_image,torch.Tensor):
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
                        ip_adapter_scale=denoise_or_ip_sacle, lora_scale=0.8,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        height=height, width=width,
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
                del pipe
                cleanup_models(keep_clone_weights_loaded=False)
                gc.collect()
                torch.cuda.empty_cache()
                from .model_loader_utils import msdiffusion_main
                image_dual = msdiffusion_main(image_a, image_b, prompts_dual, new_width, new_height, steps, seed,
                                              img_style, char_describe, char_origin, negative_prompt, clip_vision_path,
                                              model_type, lora, lora_path, lora_scale,
                                              trigger_words, ckpt_path, repo_id, guidance,
                                              mask_threshold, start_step, controlnet_path, control_image,
                                              controlnet_scale, cfg, guidance_list, scheduler_choice)
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

class Story_Easy_Function:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"function_repo": ("STRING", {"default": ""}),
                             "function_ckpt": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                             "function_cli": ("STRING", {"default": ""}),
                             }}

    RETURN_TYPES = ("CLIP",)
    ETURN_NAMES = ("clip",)
    FUNCTION = "funcion_main"
    CATEGORY = "Storydiffusion"

    def funcion_main(self, function_repo, function_ckpt,function_cli):
        if "glm" in function_cli:
           pass
        #print("test")
        return (function_repo,)


NODE_CLASS_MAPPINGS = {
    "Storydiffusion_Model_Loader": Storydiffusion_Model_Loader,
    "Storydiffusion_Sampler": Storydiffusion_Sampler,
    "Pre_Translate_prompt": Pre_Translate_prompt,
    "Comic_Type": Comic_Type,
    "Story_Easy_Function":Story_Easy_Function
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Storydiffusion_Model_Loader": "Storydiffusion_Model_Loader",
    "Storydiffusion_Sampler": "Storydiffusion_Sampler",
    "Pre_Translate_prompt": "Pre_Translate_prompt",
    "Comic_Type": "Comic_Type",
    "Story_Easy_Function":"Story_Easy_Function"
}
