# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import os
import re
import random
import torch
import gc
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import cv2
from diffusers import ( DDIMScheduler, 
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler,
                        DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler,
                       KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler,
                        DDPMScheduler, LCMScheduler)

from .msdiffusion.models.projection import Resampler
from .msdiffusion.models.modelWrapper import MSAdapter as MSAdapterWarpper
from .utils.style_template import styles
from .utils.load_models_utils import  get_lora_dict
from .PuLID.pulid.utils import resize_numpy_image_long
from transformers import AutoModel, AutoTokenizer
from comfy.utils import common_upscale,ProgressBar
import folder_paths
from comfy.clip_vision import load as clip_load
from .utils.gradio_utils import process_original_text,character_to_dict
import json

cur_path = os.path.dirname(os.path.abspath(__file__))
photomaker_dir=os.path.join(folder_paths.models_dir, "photomaker")
base_pt = os.path.join(photomaker_dir,"pt")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

lora_get = get_lora_dict()
lora_lightning_list = lora_get["lightning_xl_lora"]


SAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis","ddim", "uni_pc", "uni_pc_bh2"]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]


def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def get_scheduler(name,scheduler_):
    scheduler = False
    if name == "euler" or name =="euler_cfg_pp":
        scheduler = EulerDiscreteScheduler()
    elif name == "euler_ancestral" or name =="euler_ancestral_cfg_pp":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "ddim":
        scheduler = DDIMScheduler()
    elif name == "ddpm":
        scheduler = DDPMScheduler()
    elif name == "dpmpp_2m":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "dpmpp_2m" and scheduler_=="karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "dpmpp_2m_sde":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "dpmpp_2m" and scheduler_=="karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "dpmpp_sde" or name == "dpmpp_sde_gpu":
        scheduler = DPMSolverSinglestepScheduler()
    elif (name == "dpmpp_sde" or name == "dpmpp_sde_gpu") and scheduler_=="karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "dpm_2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "dpm_2" and scheduler_=="karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "dpm_2_ancestral":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "dpm_2_ancestral" and scheduler_=="karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "lcm":
        scheduler = LCMScheduler()
    elif name == "lms":
        scheduler = LMSDiscreteScheduler()
    elif name == "lms" and scheduler_=="karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "uni_pc":
        scheduler = UniPCMultistepScheduler()
    else:
        scheduler = EulerDiscreteScheduler()
    return scheduler



def get_extra_function(extra_function,extra_param,photomake_ckpt_path,image,infer_mode):
    auraface=False
    use_photov2=False
    cached=False
    inject=False
    onnx_provider="gpu"
    img2img_mode = True if isinstance(image, torch.Tensor) else False
    trigger_words_dual="best"
    dual_lora_scale=1.0
    dreamo_mode="ip"

    if extra_function:
        extra_function = extra_function.strip().lower()
        if "auraface" in extra_function:
            auraface=True  
       
    
    if extra_param:
        extra_param = extra_param.strip().lower()
        if "cache" in extra_param:
            cached=True
        if "inject" in extra_param:
            inject=True
        if "cpu" in extra_param:
            onnx_provider="cpu"
        if "id" in extra_param:
            dreamo_mode="id"
        elif "style" in extra_param:
            dreamo_mode="style"
        if "[" in extra_param:
            trigger_words_param=extra_param.split("[")[1].split("]")[0]
            trigger_words_dual=trigger_words_param.split(",")[0]
            dual_lora_scale=float(trigger_words_param.split(",")[1])


    if isinstance(photomake_ckpt_path, str) and img2img_mode:
        use_photov2 = True if "v2" in photomake_ckpt_path else False

    return auraface,use_photov2,img2img_mode,cached,inject,onnx_provider,dreamo_mode,trigger_words_dual,dual_lora_scale

def extract_content_from_brackets(text):
    # 正则表达式匹配多对方括号内的内容
    return re.findall(r'\[(.*?)\]', text)

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor_to_image(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensortopil_list(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor_to_image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor_to_image(i) for i in tensor_list]
    return img_list

def tensortopil_list_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensortolist(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [nomarl_tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[nomarl_tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list



def nomarl_tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples
def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor_to_image(samples)
    return img
def nomarl_upscale_tensor(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples
    
def center_crop(img):
    width, height = img.size
    square = min(width, height)
    left = (width - square) / 2
    top = (height - square) / 2
    right = (width + square) / 2
    bottom = (height + square) / 2
    return img.crop((left, top, right, bottom))

def center_crop_s(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))


def contains_brackets(s):
    return '[' in s or ']' in s

def has_parentheses(s):
    return bool(re.search(r'\(.*?\)', s))
def extract_content_from_brackets_(text):
    # 正则表达式匹配多对圆括号内的内容
    return re.findall(r'\((.*?)\)', text)

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = phi2narry(value)
        list_in[i] = modified_value
    return list_in
def remove_punctuation_from_strings(lst):
    pattern = r"[\W]+$"  # 匹配字符串末尾的所有非单词字符
    return [re.sub(pattern, '', s) for s in lst]

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


def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[style_name])
    #print(p, "test0", n)
    return p.replace("{prompt}", positive),n

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[style_name])
    #print(p,"test1",n)
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative





def face_bbox_to_square(bbox):
    ## l, t, r, b to square l, t, r, b
    l,t,r,b = bbox
    cent_x = (l + r) / 2
    cent_y = (t + b) / 2
    w, h = r - l, b - t
    r = max(w, h) / 2

    l0 = cent_x - r
    r0 = cent_x + r
    t0 = cent_y - r
    b0 = cent_y + r

    return [l0, t0, r0, b0]



def insight_face_loader(infer_mode,use_photov2,auraface,onnx_provider="cpu",mask_repo="briaai/RMBG-1.4"):
    insightface_root_path= folder_paths.base_path
    if infer_mode=="story" and use_photov2:
        from .utils.insightface_package import FaceAnalysis2
        if auraface:
            from huggingface_hub import snapshot_download
            snapshot_download(
                "fal/AuraFace-v1",
                local_dir="models/auraface",
            )
            app_face = FaceAnalysis2(name="auraface",
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"], root=insightface_root_path,
                                     allowed_modules=['detection', 'recognition'])
        else:
            app_face = FaceAnalysis2(providers=['CUDAExecutionProvider'],
                                     allowed_modules=['detection', 'recognition'])
        app_face.prepare(ctx_id=0, det_size=(640, 640))
        pipeline_mask = None
        app_face_ = None
    elif infer_mode=="kolor_face" :
        from .kolors.models.sample_ipadapter_faceid_plus import FaceInfoGenerator
        from huggingface_hub import snapshot_download
        snapshot_download(
            'DIAMONIK7777/antelopev2',
            local_dir='models/antelopev2',
        )
        app_face = FaceInfoGenerator(root_dir=insightface_root_path)
        pipeline_mask = None
        app_face_ = None
    elif infer_mode=="story_maker":
        from insightface.app import FaceAnalysis
        from transformers import pipeline
        pipeline_mask = pipeline("image-segmentation", model=mask_repo,
                                 trust_remote_code=True)

        app_face = FaceAnalysis(name='buffalo_l', root=insightface_root_path,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_face.prepare(ctx_id=0, det_size=(640, 640))
        app_face_ = None

    elif infer_mode=="story_and_maker":
        from insightface.app import FaceAnalysis
        from transformers import pipeline
        pipeline_mask = pipeline("image-segmentation", model=mask_repo,
                                 trust_remote_code=True)
        if use_photov2:
            from .utils.insightface_package import FaceAnalysis2
            if auraface:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    "fal/AuraFace-v1",
                    local_dir="models/auraface",
                )
                app_face = FaceAnalysis2(name="auraface",
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                                            root=insightface_root_path,
                                            allowed_modules=['detection', 'recognition'])
            else:
                app_face = FaceAnalysis2(providers=['CUDAExecutionProvider'],
                                            allowed_modules=['detection', 'recognition'])
            app_face.prepare(ctx_id=0, det_size=(640, 640))
            app_face_ = FaceAnalysis(name='buffalo_l', root=insightface_root_path,
                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app_face_.prepare(ctx_id=0, det_size=(640, 640))
        else:
            app_face = FaceAnalysis(name='buffalo_l', root=insightface_root_path,
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app_face.prepare(ctx_id=0, det_size=(640, 640))
            app_face_ = None
    elif infer_mode=="flux_pulid":
        import insightface
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from insightface.app import FaceAnalysis
        from huggingface_hub import snapshot_download
        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        
        providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        app_face = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=providers)
        app_face.prepare(ctx_id=0, det_size=(640, 640))
        # face_helper
        face_helper = FaceRestoreHelper(
                            upscale_factor=1,
                            face_size=512,
                            crop_ratio=(1, 1),
                            det_model='retinaface_resnet50',
                            save_ext='png',
                            device=device,)
        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)
        app_face_ = face_helper
        handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx',providers=providers)
        handler_ante.prepare(ctx_id=0)
        pipeline_mask = handler_ante
    elif infer_mode=="infiniteyou":   
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis
         # Load face encoder
        
        app_face = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_face.prepare(ctx_id=0, det_size=(640, 640))
        app_face_ = init_recognition_model('arcface', device='cuda')
        pipeline_mask = None
    else:
        app_face = None
        pipeline_mask = None
        app_face_ = None
    return app_face,pipeline_mask,app_face_


def get_float(str_in):
    list_str=str_in.split(",")
    float_box=[float(x) for x in list_str]
    return float_box


def adjust_indices(original_indices, deleted_indices):
    """根据删除的索引列表调整原索引"""
    # 处理删除列表为空的特殊情况
    if not deleted_indices:
        return original_indices.copy()  # 直接返回原索引，无需调整
    
    deleted_sorted = sorted(deleted_indices)
    adjusted = []
    for idx in original_indices:
        # 计算偏移量：有多少个删除索引 < 当前索引
        offset = sum(1 for d in deleted_sorted if d < idx)
        new_idx = idx - offset
        # 如果原索引未被删除，则保留
        if idx not in deleted_sorted:
            adjusted.append(new_idx)
        else:
            adjusted.append(-1)  # 标记为无效
    return adjusted


def get_insight_dict(app_face,app_face_,pipeline_mask,infer_mode,use_photov2,image_list,
                     character_list_,condition_image,width, height,model=None,image_proj_model=None):
    input_id_emb_s_dict = {}
    input_id_img_s_dict = {}
    input_id_emb_un_dict = {}

    for ind, img in enumerate(image_list): # 最大只有2个ID
        if infer_mode == "story" and use_photov2:
            from .utils.insightface_package import analyze_faces
            img_ = np.array(img)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            faces = analyze_faces(app_face, img_, )
            id_embed_list = torch.from_numpy((faces[0]['embedding']))
            crop_image = img_
            uncond_id_embeddings = None
        elif infer_mode == "kolor_face":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            face_info = app_face.get_faceinfo_one_img(img)
            face_bbox_square = face_bbox_to_square(face_info["bbox"])
            crop_image = img.crop(face_bbox_square)
            crop_image = crop_image.resize((336, 336))
            face_embeds = torch.from_numpy(np.array([face_info["embedding"]]))
            id_embed_list = face_embeds.to(device, dtype=torch.float16)
            uncond_id_embeddings = None
        elif infer_mode=="story_maker":
            crop_image = pipeline_mask(img, return_mask=True).convert('RGB')  # outputs a pillow mask
            # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # crop_image.copy().save(os.path.join(folder_paths.get_output_directory(),f"{timestamp}_mask.png"))
            face_info = app_face.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            id_embed_list = sorted(face_info,key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # only use the maximum face
            
            uncond_id_embeddings = img
        elif infer_mode=="story_and_maker":
            if use_photov2:
                from .utils.insightface_package import analyze_faces
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                faces = analyze_faces(app_face, img, )
                id_embed_list = torch.from_numpy((faces[0]['embedding']))
                crop_image = pipeline_mask(img, return_mask=True).convert(
                    'RGB')  # outputs a pillow mask
                face_info = app_face_.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                uncond_id_embeddings = sorted(face_info,key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # only use the maximum face
                #photomake_mode = "v2"
                # make+v2模式下，emb存v2的向量，corp 和 unemb 存make的向量
            else:  # V1不需要调用emb
                crop_image = pipeline_mask(img, return_mask=True).convert(
                    'RGB')  # outputs a pillow mask
                face_info = app_face.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                id_embed_list = sorted(face_info,key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # only use the maximum face
                uncond_id_embeddings = img

        elif infer_mode=="flux_pulid":
            id_image = resize_numpy_image_long(img, 1024)
            use_true_cfg = abs(1.0 - 1.0) > 1e-2
           
            id_embed_list, uncond_id_embeddings=model.pulid_model.get_id_embedding_( id_image,app_face_,app_face,pipeline_mask,cal_uncond=use_true_cfg)


            #id_embed_list, uncond_id_embeddings = pipe.pulid_model.get_id_embedding(id_image,cal_uncond=use_true_cfg)
            crop_image = img

        elif infer_mode=="infiniteyou":
    
            def _detect_face(app_face, id_image_cv2):
                face_info = app_face.get(id_image_cv2)
                if face_info:
                    return face_info
                else:
                    print("No face detected in the input ID image")
                    return []
            from .pipelines.pipeline_infu_flux import extract_arcface_bgr_embedding,resize_and_pad_image,draw_kps
             # Extract ID embeddings
            print('Preparing ID embeddings')
            id_image_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            face_info = _detect_face(app_face,id_image_cv2)
            if len(face_info) == 0:
                raise ValueError('No face detected in the input ID image')
            
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            landmark = face_info['kps']
            id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, app_face_)
            id_embed = id_embed.clone().unsqueeze(0).float().cuda()
            id_embed = id_embed.reshape([1, -1, 512])
            id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
            with torch.no_grad():
                id_embed = image_proj_model(id_embed)
                bs_embed, seq_len, _ = id_embed.shape
                id_embed = id_embed.repeat(1, 1, 1)
                id_embed = id_embed.view(bs_embed * 1, seq_len, -1)
                id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
            
            # Load control image
            print('Preparing the control image')
            if isinstance(condition_image, torch.Tensor):
                e1, _, _, _ = condition_image.size()
                if e1 == 1:
                    cn_image_load = [nomarl_upscale(condition_image, width, height)]
                else:
                    img_list = list(torch.chunk(condition_image, chunks=e1))
                    cn_image_load = [nomarl_upscale(img, width, height) for img in img_list]
                # control_image = control_image.convert("RGB")
                # control_image = resize_and_pad_image(control_image, (width, height))
                control_image=[] #如果是单人多张控制图，可能导致失序，所以统一改成列表，并在采样的时候依次调用
                for img in cn_image_load:
                    face_info = _detect_face(app_face,cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)) 
                    if len(face_info) == 0:
                        print('No face detected in the control image,use empty image,无法识别到面部,加载全黑图片替代.') #卡通人物很难识别，所以避免反复加载， 直接用黑图
                        face_cn=Image.fromarray(np.zeros([height, width, 3]).astype(np.uint8))
                    else:
                        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
                        face_cn = draw_kps(img, face_info['kps'])
                    control_image.append(face_cn)
            else:
                out_img = np.zeros([height, width, 3])
                control_image = Image.fromarray(out_img.astype(np.uint8))
            id_embed_list=id_embed
            crop_image = control_image  # inf use crop to control img
            uncond_id_embeddings = None 
        else:
            id_embed_list = None
            uncond_id_embeddings = None
            crop_image = None
        input_id_img_s_dict[character_list_[ind]] = [crop_image]
        input_id_emb_s_dict[character_list_[ind]] = [id_embed_list]
        input_id_emb_un_dict[character_list_[ind]] = [uncond_id_embeddings]
    
    
    app_face = None
    app_face_= None
    pipeline_mask =None
    gc_cleanup()
    
    if isinstance(condition_image, torch.Tensor) and infer_mode=="story_maker":
        e1, _, _, _ = condition_image.size()
        if e1 == 1:
            cn_image_load = [nomarl_upscale(condition_image, width, height)]
        else:
            img_list = list(torch.chunk(condition_image, chunks=e1))
            cn_image_load = [nomarl_upscale(img, width, height) for img in img_list]
        input_id_cloth_dict = {}
        if len(cn_image_load)>2:
            cn_image_load_role=cn_image_load[0:2]
        else:
            cn_image_load_role=cn_image_load
        for ind, img in enumerate(cn_image_load_role):
            input_id_cloth_dict[character_list_[ind]] = [img]
        if len(cn_image_load)>2: #处理多张control img
            my_list=cn_image_load[2:]
            for ind,img in enumerate(my_list):
                input_id_cloth_dict[f"dual{ind}"] = [img]
    else:
        input_id_cloth_dict = {}
    return input_id_emb_s_dict,input_id_img_s_dict,input_id_emb_un_dict,input_id_cloth_dict


def load_model_tag(repo,device,select_method):
    if "flor" in select_method.lower():#"thwri/CogFlorence-2-Large-Freeze"
        #pip install flash_attn
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(
            device)
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        processor = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)#tokenizer
    model.eval()
    return model,processor

class StoryLiteTag:
    def __init__(self, device,temperature,select_method,repo="pzc163/MiniCPMv2_6-prompt-generator",):
        self.device = device
        self.repo = repo
        self.select_method=select_method
        self.model, self.processor=load_model_tag(self.repo, self.device,self.select_method)
        self.temperature=temperature
    def run_tag(self,image):
        if "flor" in self.select_method.lower():
            inputs = self.processor(text="<MORE_DETAILED_CAPTION>" , images=image, return_tensors="pt").to(device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=True
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>" ,
                                                              image_size=(image.width, image.height))
            res=parsed_answer["<MORE_DETAILED_CAPTION>"]
        else:
            question = 'Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,remind you only need respons prompt itself and no other information.'
            msgs = [{'role': 'user', 'content': [image, question]}]
            res = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.processor,# tokenizer
                temperature=self.temperature
            )
            res=res.split(":",1)[1].strip('"')
        s=res.strip()
        res=re.sub(r'^\n+|\n+$', '', s)
        res.strip("'")
        logging.info(f"{res}")
        return res


def images_generator(img_list: list, ):
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
            raise "unsupport image list,must be pil,cv2 or tensor!!!"
    
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image


def load_images_list(img_list: list, ):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images
    

def fitter_cf_model_type(model):
    VALUE_=model.model.model_type.value
    if VALUE_ == 8:
        cf_model_type = "FLUX"
    elif VALUE_ == 4:
        cf_model_type = "CASCADE"
    elif VALUE_ == 6:
        cf_model_type = "FLOW"
    elif VALUE_ == 5:
        cf_model_type = "EDM"
    elif VALUE_ == 1:
        cf_model_type = "EPS"
    elif VALUE_ == 2:
        cf_model_type = "V_PREDICTION"
    elif VALUE_ == 3:
        cf_model_type = "V_PREDICTION_EDM"
    elif VALUE_ == 7:
        cf_model_type = "PREDICTION_CONTINUOUS"
    else:
         raise "unsupport checkpoints"
    return cf_model_type

def pre_text2infer(role_text,scene_text,lora_trigger_words,use_lora,tag_list):
    '''
    
    Args:
        role_index_dict: {'[Taylor]': [0, 1], '[Lecun]': [2, 3]} 
        invert_role_index_dict{0: ['[Taylor]'], 1: ['[Taylor]'], 2: ['[Lecun]'], 3: ['[Lecun]']}
        ref_role_index_dict {'[Taylor]': [0, 1], '[Lecun]': [2, 3]} 
        ref_role_totals[0, 1, 2, 3]
        role_list ['[Taylor]', '[Lecun]']
        role_dict {'[Taylor]': ' a woman img, wearing a white T-shirt, blue loose hair.', '[Lecun]': ' a man img,wearing a suit,black hair.'} 
        nc_txt_list [' a panda'] 
        nc_indexs[4]
        positions_index_dual[]
        positions_index_char_1 0 2 
        positions_index_char_2 [] 
        prompts_dual[] #['[A] play whith [B] in the garden'] 
        index_char_1_list [0, 1] 
        index_char_2_list[2, 3]

    Returns:
      character_index_dict:{'[Taylor]': [0, 3], '[sam]': [1, 2]},if 1 role {'[Taylor]': [0, 1, 2]}
      
    '''
    add_trigger_words = " " + lora_trigger_words + " style "
    # pre role_text
    role_dict, role_list = character_to_dict(role_text, use_lora, add_trigger_words)
    
    id_len=len(role_dict)
    #pre scene_text
    scene_text_origin = [i.strip() for i in scene_text.splitlines()]
    scene_text_origin = [i for i in scene_text_origin if '[' in i]  # 删除空行
    prompt_sence = [i for i in scene_text_origin if not len(extract_content_from_brackets(i)) >= 2]  # 剔除双角色的场景词
    
    positions_index_dual = [index for index, prompt in enumerate(scene_text_origin) if
                      len(extract_content_from_brackets(prompt)) >= 2]   #获取双角色出现的位置
    prompts_dual = [prompt for prompt in scene_text_origin if len(extract_content_from_brackets(prompt)) >= 2] # 改成单句中双方括号方法，利于MS组句，[A]... [B]...[C]
    
    #print(id_len,role_list)
    if id_len == 2:
        index_char_1_list=[index for index, prompt in enumerate(scene_text_origin) if role_list[0] in prompt]
        positions_index_char_1 = index_char_1_list[0]  # 获取角色出现的索引列表，并获取首次出现的位置
        index_char_2_list=[index for index, prompt in enumerate(scene_text_origin) if role_list[1] in prompt]
        positions_index_char_2 = index_char_2_list[0]  # 获取角色出现的索引列表，并获取首次出现的位置
    
       # print(positions_index_char_1, positions_index_char_2) #0,2
    else:
        index_char_1_list=[index for index, prompt in enumerate(scene_text_origin) if role_list[0] in prompt]
        index_char_2_list=[]
        positions_index_char_1=[]
        positions_index_char_2=[]

    if index_char_1_list and index_char_2_list: #dual 情况下，该列表有误，须排除重复元素
        common_ = set(index_char_1_list) & set(index_char_2_list)
        if common_:
            index_char_1_list=[x for x in index_char_1_list if x not in common_]
            index_char_2_list = [x for x in index_char_2_list if x not in common_]


    clipped_prompts = prompt_sence[:]  # copy
   
    nc_indexs = []
    for ind, prompt in enumerate(clipped_prompts): #获取NC的index
        if "[NC]" in prompt:
            nc_indexs.append(ind)
            if ind < id_len:
                raise f"The first [role] row need be a id prompts, cannot use [NC]!"
    prompts = [
        i if "[NC]" not in i else i.replace("[NC]", "") for i in clipped_prompts]
    #去除#
    prompts = [prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts]
    
    # character_dict:{'[Taylor]': ' a woman img, wearing a white T-shirt, blue loose hair.'},character_list:['[Taylor]'] 1role
    #character_dict:{'[Taylor]': ' a woman img, wearing a white T-shirt, blue loose hair.', '[Lecun]': ' a man img,wearing a suit,black hair.'},character_list:['[Taylor]', '[Lecun]'] 2 role
    #获取字典，实际用词等
    role_index_dict, invert_role_index_dict, replace_prompts, ref_role_index_dict, ref_role_totals= process_original_text(role_dict, prompts)
    
    if tag_list: #tag方法
        if len(tag_list) < len(replace_prompts):
            raise "The number of input condition images is less than the number of scene prompts!"
        replace_prompts = [prompt + " " + tag_list[i] for i, prompt in enumerate(replace_prompts)]
    
    if nc_indexs:
        for x in nc_indexs:  # 获取NC列表
            nc_txt_list = [item for i, item in enumerate(replace_prompts) if i == x]
        
        for x in nc_indexs:  # 去除NC列表
            replace_prompts = [item for i, item in enumerate(replace_prompts) if i != x]
    else:
        nc_txt_list = []
        
   
    replace_prompts=[i for i in replace_prompts]
    
    return replace_prompts,role_index_dict,invert_role_index_dict,ref_role_index_dict,ref_role_totals,role_list,role_dict,nc_txt_list,nc_indexs,positions_index_char_1,positions_index_char_2,positions_index_dual,prompts_dual,index_char_1_list,index_char_2_list


def convert_cf2diffuser(model):
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
    from diffusers import UNet2DConditionModel
    config_file = os.path.join(cur_path,"local_repo/unet/config.json")
    cf_state_dict = model.diffusion_model.state_dict()
    unet_state_dict = model.model_config.process_unet_state_dict_for_saving(cf_state_dict)
    unet_config = UNet2DConditionModel.load_config(config_file)
    Unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    cf_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, Unet.config)
    Unet.load_state_dict(cf_state_dict, strict=False)
    del cf_state_dict,unet_state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return Unet


def cf_clip(txt_list, clip, infer_mode,role_list,input_split=True):
   
    if infer_mode == "classic":
        input_split= False
    if len(role_list)==1:
        input_split= False
    if input_split:
        role_emb_dict={}
        for role,role_text_list in zip(role_list,txt_list):
            pos_cond_list=[]
            for j in role_text_list:
                tokens_p = clip.tokenize(j)
                output_p = clip.encode_from_tokens(tokens_p, return_dict=True)  # {"pooled_output":tensor}
                cond_p = output_p.pop("cond")
                if cond_p.shape[1] / 77 > 1 and infer_mode != "classic":
                # logging.warning("prompt'tokens length is abvoe 77,split it")
                    cond_p = torch.chunk(cond_p, cond_p.shape[1] // 77, dim=1)[0]
                positive = [cond_p, output_p]
                pos_cond_list.append(positive)
            role_emb_dict[role]=pos_cond_list
        return role_emb_dict
    
    pos_cond_list = []
    for i in txt_list[0]:
        tokens_p = clip.tokenize(i)
        output_p = clip.encode_from_tokens(tokens_p, return_dict=True)  # {"pooled_output":tensor}
        cond_p = output_p.pop("cond")
        if cond_p.shape[1] / 77 > 1 and infer_mode != "classic":
            # logging.warning("prompt'tokens length is abvoe 77,split it")
            cond_p = torch.chunk(cond_p, cond_p.shape[1] // 77, dim=1)[0]
        if infer_mode == "classic":
            positive = [[cond_p, output_p]]
        else:
            positive = [cond_p, output_p]
        # logging.info(f"sampler text is {i}")
        pos_cond_list.append(positive)
    return pos_cond_list

def get_eot_idx_cf(tokenizer, prompt):
    words = prompt.split()
    start = 1
    for w in words:
        start += len(tokenizer.encode(w)) - 2
    return start

def get_phrase_idx_cf(tokenizer, phrase, prompt, get_last_word=False, num=0):
    def is_equal_words(pr_words, ph_words):
        if len(pr_words) != len(ph_words):
            return False
        for pr_word, ph_word in zip(pr_words, ph_words):
            if "-"+ph_word not in pr_word and ph_word != re.sub(r'[.!?,:]$', '', pr_word):
                return False
        return True

    phrase_words = phrase.split()
    if len(phrase_words) == 0:
        return [0, 0], None
    if get_last_word:
        phrase_words = phrase_words[-1:]
    # prompt_words = re.findall(r'\b[\w\'-]+\b', prompt)
    prompt_words = prompt.split()
    start = 1
    end = 0
    res_words = phrase_words
    for i in range(len(prompt_words)):
        if is_equal_words(prompt_words[i:i+len(phrase_words)], phrase_words):
            if num != 0:
                # skip this one
                num -= 1
                continue
            end = start
            res_words = prompt_words[i:i+len(phrase_words)]
            res_words = [re.sub(r'[.!?,:]$', '', w) for w in res_words]
            prompt_words[i+len(phrase_words)-1] = res_words[-1]  # remove the last punctuation
            for j in range(i, i+len(phrase_words)):
                end += len(tokenizer.encode(prompt_words[j])) - 2
            break
        else:
            start += len(tokenizer.encode(prompt_words[i])) - 2

    if end == 0:
        return [0, 0], None

    return [start, end], res_words

def get_phrases_idx_cf(tokenizer, phrases, prompt):
    res = []
    phrase_cnt = {}
    for phrase in phrases:
        if phrase in phrase_cnt:
            cur_cnt = phrase_cnt[phrase]
            phrase_cnt[phrase] += 1
        else:
            cur_cnt = 0
            phrase_cnt[phrase] = 1
        res.append(get_phrase_idx_cf(tokenizer, phrase, prompt, num=cur_cnt)[0])
    return res

def replicate_data_by_indices(data_list, index_list1, index_list2):
    # 确定新列表的长度
    max_index = max(max(index_list1), max(index_list2)) if (index_list1 and index_list2) else 0
    new_list = [None] * (max_index + 1)
    
    # 根据索引填充数据
    for idx in index_list1:
        new_list[idx] = data_list[0]
    for idx in index_list2:
        new_list[idx] = data_list[1]
    
    return new_list



def get_ms_phrase_emb(boxes, device, weight_dtype, drop_grounding_tokens, bsz, phrase_idxes,
                      num_samples, eot_idxes,phrases,clip,tokenizer):
    cross_attention_kwargs = None
    grounding_kwargs = None
    if boxes is not None:
        boxes = torch.tensor(boxes).to(device, weight_dtype) #torch.Size([1, 2, 4])
        if phrases is not None:
            drop_grounding_tokens = drop_grounding_tokens if drop_grounding_tokens is not None else [0] * bsz
            batch_boxes = boxes.view(bsz * boxes.shape[1], -1).to(device)
            phrase_input_ids=[]
            for phrase in phrases:
                phrase_input_id = tokenizer(phrase, max_length=tokenizer.model_max_length,
                                                     padding="max_length", truncation=True,
                                                     return_tensors="pt").input_ids
                int_list = phrase_input_id.tolist()[0]
                clean_input_ids_1=[(i,1.0) for i in int_list]
                clean_input_ids_2=[(i,1.0) for i in int_list if i==int_list[0] or i==49407 ]
                clean_input_ids_2=clean_input_ids_2[:77] if len(clean_input_ids_2) >=77 else clean_input_ids_2 + [(0,1.0)]*(77 - len(clean_input_ids_2))
                phrase_input_ids.append([clean_input_ids_1,clean_input_ids_2])

            phrase_embeds_list = []
            for i in phrase_input_ids:
                _, pooled_prompt_embed_= clip.cond_stage_model.clip_l.encode_token_weights(i)
                pooled_prompt_embed_= pooled_prompt_embed_.to(device,dtype=weight_dtype)
                phrase_embeds_list.append(pooled_prompt_embed_)
                #phrase_input_id = clip.tokenize(phrase, return_word_ids=False)["l"]
                #output_=clip.cond_stage_model.clip_l.encode_token_weights(phrase_input_id)[1]
               
            phrase_embeds=torch.cat(phrase_embeds_list,dim=0)
            #print(phrase_embeds.shape,phrase_embeds.is_cuda,batch_boxes.shape,batch_boxes.is_cuda)#torch.Size([2, 768]) torch.Size([2 4])
            grounding_kwargs = {"boxes": batch_boxes, "phrase_embeds": phrase_embeds,"drop_grounding_tokens": drop_grounding_tokens}
        else:
            grounding_kwargs = None
        boxes = torch.repeat_interleave(boxes, repeats=num_samples, dim=0)
        uncond_boxes = torch.zeros_like(boxes)
        boxes = torch.cat([uncond_boxes, boxes], dim=0)
        cross_attention_kwargs = {"boxes": boxes}
    
    if phrase_idxes is not None:
        # phrase_idxes = torch.tensor(phrase_idxes).to(device, torch.int)
        # eot_idxes = torch.tensor(eot_idxes).to(device, torch.int)
        phrase_idxes = torch.tensor(phrase_idxes).to(device, weight_dtype)
        eot_idxes = torch.tensor(eot_idxes).to(device, weight_dtype)
        
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
    
    return cross_attention_kwargs,grounding_kwargs
def Infer_MSdiffusion(model, photomake_or_ipadapter_path,image_embeds,  prompt_embeds_, negative_prompt_embeds_,grounding_kwargs,cross_attention_kwargs, bsz,mask_threshold,height,width,steps,seed,cfg,pooled_prompt_embeds,negative_pooled_prompt_embeds,):
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        embedding_dim=1664,
        output_dim=model.unet.config.cross_attention_dim,
        ff_mult=4,
        latent_init_mode="grounding",
        phrase_embeddings_dim=768,
    ).to(device, dtype=torch.float16)
    PIPE = MSAdapterWarpper(model, image_proj_model, ckpt_path=photomake_or_ipadapter_path, device=device, num_tokens=16)
    samples=PIPE.generate(model, image_embeds,  prompt_embeds_, negative_prompt_embeds_,grounding_kwargs,cross_attention_kwargs, bsz,height,width,steps,seed,cfg,pooled_prompt_embeds,negative_pooled_prompt_embeds,scale=1.0,
                 num_samples=bsz,  weight_dtype=torch.float16, subject_scales=None, mask_threshold=mask_threshold, start_step=5)
    return samples

def convert_clip2diffuser(clip):
   
    from transformers import (
            CLIPTextModel,CLIPTextModelWithProjection,
            CLIPTextConfig,
                                            )
    from contextlib import nullcontext

    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        is_accelerate_available = True
    except:
        pass
    
    import comfy.model_management as mm
   
    offload_device=mm.unet_offload_device()
   
    text_encoder_config=CLIPTextConfig.from_pretrained(os.path.join(cur_path, 'local_repo/text_encoder'), local_files_only=True)
    ctx = init_empty_weights if is_accelerate_available else nullcontext
    with ctx():
        text_encoder = CLIPTextModel(text_encoder_config)
    text_encoder_sd = clip.get_sd()
    text_encoder_2_sd = clip.get_sd()

    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_open_clip_checkpoint,convert_ldm_clip_checkpoint
    text_encoder = convert_ldm_clip_checkpoint(text_encoder_sd, text_encoder=text_encoder)
    text_encoder=None
    text_encoder_2 = convert_open_clip_checkpoint(text_encoder_2_sd,config_name=os.path.join(cur_path, 'local_repo/text_encoder_2'))    
   
    return text_encoder,text_encoder_2
    


def convert_cfvae2diffuser(VAE,use_flux=False):
    from diffusers import AutoencoderKL
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint
    if use_flux:
        vae_config = os.path.join(cur_path, "config/FLUX.1-dev/vae/config.json")
    else:
        vae_config = os.path.join(cur_path, "local_repo/vae/config.json")
    vae_state_dict=VAE.get_sd()
    ae_config = AutoencoderKL.load_config(vae_config)
    
    if not use_flux:
        AE = AutoencoderKL.from_config(ae_config).to(device, torch.float16)
        cf_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, AE.config)
        AE.load_state_dict(cf_state_dict, strict=False)
        del cf_state_dict,vae_state_dict
    else:
        AE = AutoencoderKL.from_config(ae_config).to(device, torch.bfloat16)
        AE.load_state_dict(vae_state_dict, strict=False)
        del vae_state_dict
    torch.cuda.empty_cache()
    return AE


def Loader_storydiffusion(cf_model,photomake_ckpt_path,VAE,UNET=None):
    sdxl_repo = os.path.join(cur_path, "local_repo")
    vae_config=OmegaConf.load(os.path.join(cur_path,"local_repo/vae/config.json"))
    
    from .utils.pipeline_wrapper import StableDiffusionXLPipelineWrapper
    from .utils.pipeline_img import PhotoMakerStableDiffusionXLPipeline as PhotoMakerStableDiffusionXLPipelineWrapper
    from .utils.pipeline_v2_warrper import  PhotoMakerStableDiffusionXLPipeline as PhotoMakerStableDiffusionXLPipelineV2
    if cf_model is not None:
        Unet=convert_cf2diffuser(cf_model.model)
        if photomake_ckpt_path is not None: # guess img2img
            if VAE is None:
                raise "need link a SDXL vae when use img2img!"
            vae = convert_cfvae2diffuser(VAE)
            if "v1" in photomake_ckpt_path:
                model = PhotoMakerStableDiffusionXLPipelineWrapper.from_pretrained(sdxl_repo, unet=Unet, vae=vae,text_encoder=None,text_encoder_2=None,
                                                                     torch_dtype=torch.float16, use_safetensors=True)
                model.load_photomaker_adapter(
                photomake_ckpt_path,
                subfolder="",
                weight_name="photomaker-v1.bin",
                trigger_word="img"  # define the trigger word
            )
            
            else:
                model = PhotoMakerStableDiffusionXLPipelineV2.from_pretrained(sdxl_repo, unet=Unet, vae=vae,text_encoder=None,text_encoder_2=None,
                                                                     torch_dtype=torch.float16, use_safetensors=True)
                model.load_photomaker_adapter(
                photomake_ckpt_path,
                subfolder="",
                weight_name="photomaker-v2.bin",
                trigger_word="img",
                pm_version='v2',
            )
        else:
            model = StableDiffusionXLPipelineWrapper.from_pretrained(sdxl_repo, unet=Unet, vae_config=vae_config,
                                                                     torch_dtype=torch.float16, use_safetensors=True)
    elif UNET is not None:  
         model = StableDiffusionXLPipelineWrapper.from_pretrained(sdxl_repo, unet=UNET.unet, vae_config=vae_config,
                                                                     torch_dtype=torch.float16, use_safetensors=True)     
    else:
        raise "need  link a comfyUI checkpoint model!"

    model.to(device)
    return model




def Loader_story_maker(cf_model,ipadapter_ckpt_path,VAE,low_vram,lora_scale,controlnet=None,UNET=None):
    sdxl_repo = os.path.join(cur_path, "local_repo")
    #vae_config = OmegaConf.load(vae_config)
    if cf_model is not None:
        Unet = convert_cf2diffuser(cf_model.model)
        AE=convert_cfvae2diffuser(VAE)
        from .StoryMaker.pipeline_sdxl_storymaker_wrapper import StableDiffusionXLStoryMakerPipeline as StableDiffusionXLStoryMakerPipeline_wapper
        model = StableDiffusionXLStoryMakerPipeline_wapper.from_pretrained(sdxl_repo, vae=AE,unet=Unet,text_encoder=None,text_encoder_2=None, 
                                                                 torch_dtype=torch.float16, use_safetensors=True)
    elif UNET is not None:
        AE=convert_cfvae2diffuser(VAE)
        from .StoryMaker.pipeline_sdxl_storymaker_wrapper import StableDiffusionXLStoryMakerPipeline as StableDiffusionXLStoryMakerPipeline_wapper
        model = StableDiffusionXLStoryMakerPipeline_wapper.from_pretrained(sdxl_repo, vae=AE,unet=UNET.unet,text_encoder=None,text_encoder_2=None,
                                                                               torch_dtype=torch.float16, use_safetensors=True)
    else:
        raise "need  link a comfyUI checkpoint model!"
       
    
    if device != "mps":
        if not low_vram:
            model.cuda()
    
    model.load_storymaker_adapter(ipadapter_ckpt_path, scale=0.8, lora_scale=lora_scale, controlnet=controlnet)
    #model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    return model


def Loader_Flux_Pulid(model,cf_model, ipadapter_ckpt_path,quantized_mode,aggressive_offload,offload,if_repo,clip_vision_path):
    
    logging.info("start flux-pulid processing...")
    from .PuLID.app_flux import FluxGenerator
    
    pipe = FluxGenerator(model, "cuda:0", offload,aggressive_offload, pretrained_model=ipadapter_ckpt_path,
                            quantized_mode=quantized_mode, if_repo=if_repo,clip_vision_path=clip_vision_path)

    return pipe

def Loader_InfiniteYou(extra_info,VAE,quantize_mode):
    logging.info("start InfiniteYou mode processing...")    
    from .pipelines.pipeline_infu_flux import InfUFluxPipeline
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig,FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig,T5EncoderModel
    from .pipelines.resampler import Resampler
    
    
    # quantize T5 
    # quant_config = TransformersBitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #     )

    # text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    #         repo_id,
    #         subfolder="text_encoder_2",
    #         quantization_config=quant_config,
    #         torch_dtype=torch.bfloat16,
    #     ) # init nf4 
    repo_list=[i for i  in [extra_info.get("repo1_path") , extra_info.get("repo2_path")] if i is not None]
    assert repo_list is not [] ,"repo_list is None"
    find_infusenet=[i for i in repo_list if "sim_stage1" in i or "aes_stage2" in i]

    assert len(find_infusenet)>=1 ,"no infusenet"

    repo_id=os.path.join(cur_path, "config/FLUX.1-dev")
    use_svdq=False
    if extra_info.get("svdq_repo") is not None:
        print("use svdq quantization")   
        from nunchaku import NunchakuFluxTransformer2dModel
        use_svdq=True
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(extra_info.get("svdq_repo"),offload=True)
        try:
            transformer.set_attention_impl("nunchaku-fp16")
        except:
            pass
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
    elif extra_info.get("gguf_path") is not None:
        print("use gguf quantization")   
        from diffusers import  GGUFQuantizationConfig
        transformer = FluxTransformer2DModel.from_single_file(
            extra_info.get("gguf_path"),
            config=os.path.join(repo_id, "transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
    elif extra_info.get("unet_path") is not None:

        if quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_single_file(extra_info.get("unet_path"), config=os.path.join(repo_id,"transformer/config.json"),
                                                                            torch_dtype=torch.bfloat16)
        else:
            from safetensors.torch import load_file
            t_state_dict=load_file(extra_info.get("unet_path"))
            unet_config = FluxTransformer2DModel.load_config(os.path.join(repo_id,"transformer/config.json"))
            transformer = FluxTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
            transformer.load_state_dict(t_state_dict, strict=False)
            del t_state_dict
            gc_cleanup()
    else:
        vae = None
        find_flux=[i for i in repo_list if "flux" in i ]
        if not find_flux:
            raise "need fill flux repo  in EasyFunction_Lite repo1 or repo2!"
        print("use nf4 quantization")   
        if quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_pretrained(
                find_flux[0],
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True,),
                torch_dtype=torch.bfloat16,
            )
                   
        elif quantize_mode=="nf4": #nf4
            transformer = FluxTransformer2DModel.from_pretrained(
                find_flux[0],
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                ),
                torch_dtype=torch.bfloat16,)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                find_flux[0],
                subfolder="transformer",
                 torch_dtype=torch.bfloat16,
                )

        
    infusenet_path = os.path.join(find_infusenet[0], 'InfuseNetModel')

     
    pipe = InfUFluxPipeline(
            repo_id,infusenet_path,
            text_encoder_2=None,
            text_encoder=None,
            transformer=transformer,
            vae=vae,
            use_svdq=use_svdq,
        )
    # Load image proj model
    num_tokens = 8 # image_proj_num_tokens
    image_emb_dim = 512
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_emb_dim,
        output_dim=4096,
        ff_mult=4,
    )
    image_proj_model_path = os.path.join(find_infusenet[0], 'image_proj_model.bin')
    ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu",weights_only=False)
    image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
    del ipm_state_dict
    torch.cuda.empty_cache()
    image_proj_model.to('cuda', torch.bfloat16)
    image_proj_model.eval()

    pipe.pipe.enable_model_cpu_offload()
    return pipe,image_proj_model

def load_pipeline_consistory(cf_model,VAE):
    from .consistory.consistory_unet_sdxl import ConsistorySDXLUNet2DConditionModel
    from .consistory.consistory_pipeline_wapper import ConsistoryExtendAttnSDXLPipeline as ConsistoryExtendAttnSDXLPipeline_wapper
    float_type = torch.float16
    
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if cf_model is not None:
        config_file = os.path.join(cur_path,"local_repo/unet/config.json")
        sdxl_repo = os.path.join(cur_path, "local_repo")

        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
        cf_state_dict = cf_model.model.diffusion_model.state_dict()
        unet_state_dict = cf_model.model.model_config.process_unet_state_dict_for_saving(cf_state_dict)

        unet_config = ConsistorySDXLUNet2DConditionModel.load_config(config_file)
        Unet = ConsistorySDXLUNet2DConditionModel.from_config(unet_config).to(torch.float16)
        cf_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, Unet.config)
        Unet.load_state_dict(cf_state_dict, strict=False)
        del cf_state_dict,unet_state_dict
        gc_cleanup()
        
        AE=convert_cfvae2diffuser(VAE)
       
        story_pipeline=ConsistoryExtendAttnSDXLPipeline_wapper.from_pretrained(sdxl_repo,unet=Unet,vae=AE,text_encoder=None,text_encoder_2=None, torch_dtype=float_type, )
        story_pipeline.to(device)
    else:
        raise "need link a sdxl checkpoints"
    
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    return story_pipeline

def Loader_KOLOR(repo_id,clip_vision_path,face_ckpt,):

    from .kolors.models.modeling_chatglm import ChatGLMModel
    from .kolors.models.tokenization_chatglm import ChatGLMTokenizer
    from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
    from diffusers import  EulerDiscreteScheduler, UNet2DConditionModel, AutoencoderKL

    logging.info("loader kolor processing...")

    # text_encoder = ChatGLMModel.from_pretrained(
    #     f'{repo_id}/text_encoder', torch_dtype=torch.float16).half()
    vae = AutoencoderKL.from_pretrained(f"{repo_id}/vae", revision=None).half()
    #tokenizer = ChatGLMTokenizer.from_pretrained(f'{repo_id}/text_encoder')
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{repo_id}/scheduler")

   
    from .kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter_FaceID import \
        StableDiffusionXLPipeline as StableDiffusionXLPipelineFaceID
    unet = UNet2DConditionModel.from_pretrained(f'{repo_id}/unet', revision=None).half()
    
    if clip_vision_path:

        clip_image_encoder = clip_vision_path
        clip_image_processor = CLIPImageProcessor(size=224, crop_size=224)
        use_singel_clip = True
    else:
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            f'{repo_id}/clip-vit-large-patch14-336', ignore_mismatched_sizes=True)
        clip_image_encoder.to("cuda")
        clip_image_processor = CLIPImageProcessor(size=336, crop_size=336)
        use_singel_clip = False
    
    pipe = StableDiffusionXLPipelineFaceID(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        scheduler=scheduler,
        face_clip_encoder=clip_image_encoder,
        face_clip_processor=clip_image_processor,
        force_zeros_for_empty_prompt=False,
        use_single_clip=use_singel_clip,
    )
    pipe = pipe.to("cuda")
    pipe.load_ip_adapter_faceid_plus(face_ckpt, device="cuda")
    pipe.set_face_fidelity_scale(0.8)
    return pipe

def GLM_clip(cur_path,chatglm3_path): #https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/blob/main/nodes.py @kijai
    from .kolors.models.modeling_chatglm import ChatGLMModel, ChatGLMConfig
    from contextlib import nullcontext
    from comfy.utils import ProgressBar, load_torch_file
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        is_accelerate_available = True
    except:
        pass
    
    import comfy.model_management as mm
    import json
    offload_device=mm.unet_offload_device()
    text_encoder_config = os.path.join(cur_path, 'config/glm/text_encoder_config.json')
    with open(text_encoder_config, 'r') as file:
        config = json.load(file)

    text_encoder_config = ChatGLMConfig(**config)
    with (init_empty_weights() if is_accelerate_available else nullcontext()):
        text_encoder = ChatGLMModel(text_encoder_config)
        if '4bit' in chatglm3_path:
            text_encoder.quantize(4)
        elif '8bit' in chatglm3_path:
            text_encoder.quantize(8)

    text_encoder_sd = load_torch_file(chatglm3_path)
    if is_accelerate_available:
        for key in text_encoder_sd:
            set_module_tensor_to_device(text_encoder, key, device=offload_device, value=text_encoder_sd[key])
    else:
        text_encoder.load_state_dict()
    

    return text_encoder

def glm_single_encode(chatglm3_model, prompt_list,role_list, negative_prompt_, num_images_per_prompt,nc_mode=False): #https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/blob/main/nodes.py @kijai
    import comfy.model_management as mm
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    mm.unload_all_models()
    mm.soft_empty_cache()
        # Function to randomly select an option from the brackets
    # def choose_random_option(match):
    #     options = match.group(1).split('|')
    #     return random.choice(options)

    # # Randomly choose between options in brackets for prompt and negative_prompt
    # prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, prompt)
    # negative_prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, negative_prompt)

    # if "|" in prompt:
    #     prompt = prompt.split("|")
    #     negative_prompt = [negative_prompt] * len(prompt)  # Replicate negative_prompt to match length of prompt list

    do_classifier_free_guidance = True
    #print(prompt)
    if nc_mode:
        only_role_emb,only_role_emb_ne = [],[]
        for prompt,negative_prompt in zip(prompt_list,[negative_prompt_]*len(prompt_list)):
                
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)

            # Define tokenizers and text encoders
            tokenizer = chatglm3_model['tokenizer']
            text_encoder = chatglm3_model['text_encoder']

            text_encoder.to(device)

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            output = text_encoder(
                    input_ids=text_inputs['input_ids'] ,
                    attention_mask=text_inputs['attention_mask'],
                    position_ids=text_inputs['position_ids'],
                    output_hidden_states=True)
            
            prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
            text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)


            if do_classifier_free_guidance:
                uncond_tokens = []
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt
            

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                output = text_encoder(
                        input_ids=uncond_input['input_ids'] ,
                        attention_mask=uncond_input['attention_mask'],
                        position_ids=uncond_input['position_ids'],
                        output_hidden_states=True)
                negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
                negative_text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_images_per_prompt, seq_len, -1
                    )

            bs_embed = text_proj.shape[0]
            text_proj = text_proj.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            negative_text_proj = negative_text_proj.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            math_emb=[prompt_embeds,text_proj]
            only_role_emb.append(math_emb)
            math_emb_n=[negative_prompt_embeds,negative_text_proj]
            only_role_emb_ne.append(math_emb_n)
        
        text_encoder.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()
        return only_role_emb,only_role_emb_ne
    else:
        role_emb_dict={}
        for key,prompts in zip(role_list,prompt_list):
            only_role_emb,only_role_emb_ne = [],[]
            for prompt,negative_prompt in zip(prompts,[negative_prompt_]*len(prompts)):
                
                if prompt is not None and isinstance(prompt, str):
                    batch_size = 1
                elif prompt is not None and isinstance(prompt, list):
                    batch_size = len(prompt)

                # Define tokenizers and text encoders
                tokenizer = chatglm3_model['tokenizer']
                text_encoder = chatglm3_model['text_encoder']

                text_encoder.to(device)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                output = text_encoder(
                        input_ids=text_inputs['input_ids'] ,
                        attention_mask=text_inputs['attention_mask'],
                        position_ids=text_inputs['position_ids'],
                        output_hidden_states=True)
                
                prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
                text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)


                if do_classifier_free_guidance:
                    uncond_tokens = []
                    if negative_prompt is None:
                        uncond_tokens = [""] * batch_size
                    elif prompt is not None and type(prompt) is not type(negative_prompt):
                        raise TypeError(
                            f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                            f" {type(prompt)}."
                        )
                    elif isinstance(negative_prompt, str):
                        uncond_tokens = [negative_prompt]
                    elif batch_size != len(negative_prompt):
                        raise ValueError(
                            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                            " the batch size of `prompt`."
                        )
                    else:
                        uncond_tokens = negative_prompt
                

                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    output = text_encoder(
                            input_ids=uncond_input['input_ids'] ,
                            attention_mask=uncond_input['attention_mask'],
                            position_ids=uncond_input['position_ids'],
                            output_hidden_states=True)
                    negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
                    negative_text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]

                    if do_classifier_free_guidance:
                        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                        seq_len = negative_prompt_embeds.shape[1]

                        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                        negative_prompt_embeds = negative_prompt_embeds.view(
                            batch_size * num_images_per_prompt, seq_len, -1
                        )

                bs_embed = text_proj.shape[0]
                text_proj = text_proj.repeat(1, num_images_per_prompt).view(
                    bs_embed * num_images_per_prompt, -1
                )
                negative_text_proj = negative_text_proj.repeat(1, num_images_per_prompt).view(
                    bs_embed * num_images_per_prompt, -1
                )
                math_emb=[prompt_embeds,text_proj]
                only_role_emb.append(math_emb)
                math_emb_n=[negative_prompt_embeds,negative_text_proj]
                only_role_emb_ne.append(math_emb_n)
            role_emb_dict[key]=only_role_emb
    
        text_encoder.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()

        return role_emb_dict,only_role_emb_ne


def encode_prompt_with_trigger_word(clip,model_,
    prompt: str,
    prompt_2 = None,
    num_id_images= 1,
    device = None,
    prompt_embeds = None,
    pooled_prompt_embeds= None,
    class_tokens_mask = None,
    nc_flag: bool = False,
    trigger_word="img",
    unet_dtype=torch.float16
):
    device = device 
    tokenizer_2=model_.tokenizer_2
    tokenizer=model_.tokenizer
    #text_encoders=clip.encode_from_tokens

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Find the token id of the trigger word
    image_token_id = tokenizer_2.convert_tokens_to_ids(trigger_word)

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    # text_encoders = (
    #     [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
    # )

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
       
        for prompt, tokenizer,text_en in zip(prompts, tokenizers,[clip.cond_stage_model.clip_l.encode_token_weights,clip.cond_stage_model.clip_g.encode_token_weights]):
            input_ids = tokenizer.encode(prompt) # TODO: batch encode
            clean_index = 0
            clean_input_ids = []
            class_token_index = []
            # Find out the corresponding class word token based on the newly added trigger word token
            for i, token_id in enumerate(input_ids):
                if token_id == image_token_id:
                    class_token_index.append(clean_index - 1)
                else:
                    clean_input_ids.append(token_id)
                    clean_index += 1
            if nc_flag:
                return None, None, None
            if len(class_token_index) > 1:
                raise ValueError(
                    f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                        Trigger word: {trigger_word}, Prompt: {prompt}."
                )
            elif len(class_token_index) == 0 and not nc_flag:
                raise ValueError(
                    f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                        Trigger word: {trigger_word}, Prompt: {prompt}."
                )
            class_token_index = class_token_index[0]

            # Expand the class word token and corresponding mask
            class_token = clean_input_ids[class_token_index]
            clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images + \
                clean_input_ids[class_token_index+1:]

            # Truncation or padding
            max_len = tokenizer.model_max_length
            if len(clean_input_ids) > max_len:
                clean_input_ids = clean_input_ids[:max_len]
            else:
                clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                    max_len - len(clean_input_ids)
                )

            class_tokens_mask = [True if class_token_index <= i < class_token_index+num_id_images else False \
                    for i in range(len(clean_input_ids))]
            #print(clean_input_ids)
            #print(max_len)
            
            clean_input_ids=clean_input_ids[:77] if len(clean_input_ids) > 77 else clean_input_ids + [0]*(77 - len(clean_input_ids))
                
            clean_input_ids_1=[(i,1.0) for i in clean_input_ids]
            clean_input_ids_2=[(i,1.0) for i in clean_input_ids if i==clean_input_ids[0] or i==49407 ]
            clean_input_ids_2=clean_input_ids_2[:77] if len(clean_input_ids_2) > 77 else clean_input_ids_2 + [(0,1.0)]*(77 - len(clean_input_ids_2))
            clean_input_ids_list=[clean_input_ids_1,clean_input_ids_2]
            #print(clean_input_ids)
            #clean_input_ids_ = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
            class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

           
            prompt_embeds, pooled_prompt_embeds = text_en(clean_input_ids_list) # print(prompt_embeds.shape,pooled_prompt_embeds.shape)torch.Size([1, 154, 768]) torch.Size([1, 768]) torch.Size([1, 154, 1280]) torch.Size([1, 1280])


            prompt_embeds_list.append((prompt_embeds,pooled_prompt_embeds))

        #prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        #prompt_embeds=prompt_embeds_list[1]
    pooled_prompt_embeds=prompt_embeds_list[1][1]
    # cut_to = min(prompt_embeds_list[0][0].shape[1], prompt_embeds_list[1][0].shape[1])
    # print(cut_to)
    cut_to=77 
    prompt_embeds= torch.cat([prompt_embeds_list[0][0][:,:cut_to], prompt_embeds_list[1][0][:,:cut_to]], dim=-1)
    prompt_embeds = prompt_embeds.to(device=device,dtype=unet_dtype, )
    class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case

    return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

def cf_clip_single(clip,prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,):
    
    tokens_p = clip.tokenize(prompt)
    output_p = clip.encode_from_tokens(tokens_p, return_dict=True)  # {"pooled_output":tensor}
    cond_p = output_p.pop("cond")
    if cond_p.shape[1] / 77 > 1 :
        logging.warning("prompt'tokens length is abvoe 77,split it")
        cond_p = torch.chunk(cond_p, cond_p.shape[1] // 77, dim=1)[0]
    pool_p=output_p.pop("pooled_output")

    tokens_n = clip.tokenize(negative_prompt)
    output_n = clip.encode_from_tokens(tokens_n, return_dict=True)  # {"pooled_output":tensor}
    cond_n = output_n.pop("cond")
    if cond_n.shape[1] / 77 > 1 :
        logging.warning("prompt'tokens length is abvoe 77,split it")
        cond_n = torch.chunk(cond_n, cond_n.shape[1] // 77, dim=1)[0]
    pool_n = output_n.pop("pooled_output")

    return cond_p,cond_n,pool_p,pool_n # TODO: replace the pooled_prompt_embeds with text only prompt

    

def photomaker_clip(clip,model_,prompt_list,negative_prompt,input_id_images,trigger_word="img",num_images_per_prompt=1,nc_flag=False,prompt_2=None,prompt_embeds=None,pooled_prompt_embeds=None,class_tokens_mask=None,prompt_embeds_text_only=None,negative_pooled_prompt_embeds=None):
    id_encoder=model_.id_encoder
    id_image_processor=model_.id_image_processor

    tokenizer=model_.tokenizer_2
    batch_size=len(prompt_list)

    num_id_images = len(input_id_images) #双角色时，须确保输入的图片是列表，且跟prompt对应上
    prompt_arr = prompt_list
    negative_prompt_embeds_arr = []
    prompt_embeds_text_only_arr = []
    prompt_embeds_arr = []
    latents_arr = []
    add_time_ids_arr = []
    negative_pooled_prompt_embeds_arr = []
    pooled_prompt_embeds_text_only_arr = []
    pooled_prompt_embeds_arr = []
    for prompt in prompt_arr:
        (
            prompt_embeds,
            pooled_prompt_embeds,
            class_tokens_mask,
        ) = encode_prompt_with_trigger_word(clip,model_,
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_id_images=num_id_images,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            class_tokens_mask=class_tokens_mask,
            nc_flag = nc_flag,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        # encode, remove trigger word token, then decode
        tokens_text_only = tokenizer.encode(prompt, add_special_tokens=False)
        trigger_word_token = tokenizer.convert_tokens_to_ids(trigger_word)
        if not nc_flag:
            tokens_text_only.remove(trigger_word_token)
        prompt_text_only = tokenizer.decode(tokens_text_only, add_special_tokens=False)
        #print(prompt_text_only)
        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            pooled_prompt_embeds_text_only, # TODO: replace the pooled_prompt_embeds with text only prompt
            negative_pooled_prompt_embeds,
        ) = cf_clip_single(clip,
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 5. Prepare the input ID images
        dtype = next(id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype) # TODO: multiple prompts

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        if not nc_flag:
            # 6. Get the update text embedding with the stacked ID embedding
            prompt_embeds = id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            pooled_prompt_embeds_arr.append(pooled_prompt_embeds)
            pooled_prompt_embeds = None

        negative_prompt_embeds_arr.append(negative_prompt_embeds)
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds_arr.append(negative_pooled_prompt_embeds)
        negative_pooled_prompt_embeds = None
        prompt_embeds_text_only_arr.append(prompt_embeds_text_only)
        prompt_embeds_text_only = None
        prompt_embeds_arr.append(prompt_embeds)
        prompt_embeds = None
        pooled_prompt_embeds_text_only_arr.append(pooled_prompt_embeds_text_only)
        pooled_prompt_embeds_text_only = None

    emb_dict={"negative_prompt_embeds_arr":negative_prompt_embeds_arr,
              "negative_prompt_embeds":negative_prompt_embeds,
              "negative_pooled_prompt_embeds_arr":negative_pooled_prompt_embeds_arr,
              "negative_pooled_prompt_embeds":negative_pooled_prompt_embeds,
              "prompt_embeds_text_only_arr":prompt_embeds_text_only_arr,
              "prompt_embeds_text_only":prompt_embeds_text_only,
              "prompt_embeds_arr":prompt_embeds_arr,
              "prompt_embeds":prompt_embeds,
              "pooled_prompt_embeds":pooled_prompt_embeds,
              "pooled_prompt_embeds_text_only_arr":pooled_prompt_embeds_text_only_arr,
              "pooled_prompt_embeds_text_only":pooled_prompt_embeds_text_only,
              "pooled_prompt_embeds_arr":pooled_prompt_embeds_arr,
                "batch_size":batch_size,
              }    
    return emb_dict


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
def encode_prompt_with_trigger_word_v2(clip,model_,
    prompt: str,
    prompt_2 = None,
    num_id_images= 1,
    device = None,
    prompt_embeds = None,
    negative_prompt = None,
    negative_prompt_2 = None,
    negative_prompt_embeds= None,
    pooled_prompt_embeds= None,
    negative_pooled_prompt_embeds= None,
    num_images_per_prompt=1,
    class_tokens_mask = None,
    nc_flag: bool = False,
    trigger_word="img",
    unet_dtype=torch.float16,
    clip_skip=None,
    lora_scale=None,
    do_classifier_free_guidance=True,
    force_zeros_for_empty_prompts=False,
    num_tokens=2,

):
    
    device = device 
    tokenizer_2=model_.tokenizer_2
    tokenizer=model_.tokenizer
    #text_encoders=clip.encode_from_tokens

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Find the token id of the trigger word
    image_token_id = tokenizer_2.convert_tokens_to_ids(trigger_word)

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    # text_encoders = (
    #     [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
    # )

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
       
        for prompt, tokenizer,text_en in zip(prompts, tokenizers,[clip.cond_stage_model.clip_l.encode_token_weights,clip.cond_stage_model.clip_g.encode_token_weights]):
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, tokenizer)
            
            input_ids = tokenizer.encode(prompt) # TODO: batch encode
            clean_index = 0
            clean_input_ids = []
            class_token_index = []
            # Find out the corresponding class word token based on the newly added trigger word token
            for i, token_id in enumerate(input_ids):
                if token_id == image_token_id:
                    class_token_index.append(clean_index - 1)
                else:
                    clean_input_ids.append(token_id)
                    clean_index += 1
            if nc_flag:
                return None, None, None,None, None
            if len(class_token_index) > 1:
                raise ValueError(
                    f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                        Trigger word: {trigger_word}, Prompt: {prompt}."
                )
            elif len(class_token_index) == 0 and not nc_flag:
                raise ValueError(
                    f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                        Trigger word: {trigger_word}, Prompt: {prompt}."
                )
            class_token_index = class_token_index[0]

            # Expand the class word token and corresponding mask
            class_token = clean_input_ids[class_token_index]
            clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images * num_tokens + clean_input_ids[class_token_index + 1:]
            # Truncation or padding
            max_len = tokenizer.model_max_length
            if len(clean_input_ids) > max_len:
                clean_input_ids = clean_input_ids[:max_len]
            else:
                clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                    max_len - len(clean_input_ids)
                )

            #class_tokens_mask = [True if class_token_index <= i < class_token_index+num_id_images else False for i in range(len(clean_input_ids))]
            class_tokens_mask = [True if class_token_index <= i < class_token_index + (num_id_images * num_tokens) else False for i in range(len(clean_input_ids))]
            #print(clean_input_ids)
            #print(max_len)
            
            clean_input_ids=clean_input_ids[:77] if len(clean_input_ids) > 77 else clean_input_ids + [0]*(77 - len(clean_input_ids))
                
            clean_input_ids_1=[(i,1.0) for i in clean_input_ids]
            clean_input_ids_2=[(i,1.0) for i in clean_input_ids if i==clean_input_ids[0] or i==49407 ]
            clean_input_ids_2=clean_input_ids_2[:77] if len(clean_input_ids_2) > 77 else clean_input_ids_2 + [(0,1.0)]*(77 - len(clean_input_ids_2))
            clean_input_ids_list=[clean_input_ids_1,clean_input_ids_2]
            #print(clean_input_ids)
            #clean_input_ids_ = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
            class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

           
            prompt_embeds, pooled_prompt_embeds = text_en(clean_input_ids_list) # print(prompt_embeds.shape,pooled_prompt_embeds.shape)torch.Size([1, 154, 768]) torch.Size([1, 768]) torch.Size([1, 154, 1280]) torch.Size([1, 1280])


            prompt_embeds_list.append((prompt_embeds,pooled_prompt_embeds))

        #prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        #prompt_embeds=prompt_embeds_list[1]
    pooled_prompt_embeds=prompt_embeds_list[1][1]
    # cut_to = min(prompt_embeds_list[0][0].shape[1], prompt_embeds_list[1][0].shape[1])
    # print(cut_to)
    cut_to=77 
    prompt_embeds= torch.cat([prompt_embeds_list[0][0][:,:cut_to], prompt_embeds_list[1][0][:,:cut_to]], dim=-1)
    prompt_embeds = prompt_embeds.to(device=device,dtype=unet_dtype, )
    class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case
    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None and force_zeros_for_empty_prompts
    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt
        
        # normalize str to list
        #negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt = str(negative_prompt) if isinstance(negative_prompt, list) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )
        
        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        # elif batch_size != len(negative_prompt):
        #     raise ValueError(
        #         f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
        #         f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
        #         " the batch size of `prompt`."
        #     )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]
        # ng 没用上
        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, [clip.cond_stage_model.clip_l.encode_token_weights,clip.cond_stage_model.clip_g.encode_token_weights]):
            # if isinstance(self, TextualInversionLoaderMixin):
            #     negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)
            
            max_length = prompt_embeds.shape[1]
            uncond_input=tokenizer.encode(negative_prompt)
            # uncond_input = tokenizer(
            #     negative_prompt,
            #     padding="max_length",
            #     max_length=max_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            # negative_prompt_embeds = text_encoder(
            #     uncond_input.input_ids.to(device),
            #     output_hidden_states=True,
            # )
            clean_input_ids_n=uncond_input[:77] if len(uncond_input) > 77 else uncond_input + [0]*(77 - len(uncond_input))
                
            clean_input_ids_1_=[(i,1.0) for i in clean_input_ids_n]
            clean_input_ids_2_=[(i,1.0) for i in clean_input_ids_n if i==clean_input_ids_n[0] or i==49407 ]
            clean_input_ids_2_=clean_input_ids_2_[:77] if len(clean_input_ids_2_) > 77 else clean_input_ids_2_ + [(0,1.0)]*(77 - len(clean_input_ids_2_))
            clean_input_ids_list_n=[clean_input_ids_1_,clean_input_ids_2_]
            prompt_embeds_n, pooled_prompt_embeds_n = text_en(clean_input_ids_list_n)
            negative_prompt_embeds_list.append((prompt_embeds_n,pooled_prompt_embeds_n))


          
            # We are only ALWAYS interested in the pooled output of the final text encoder
            # negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            # negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            
            # negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_pooled_prompt_embeds=negative_prompt_embeds_list[1][1]
        #negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
        cut_to=77 
        negative_prompt_embeds= torch.cat([negative_prompt_embeds_list[0][0][:,:cut_to], negative_prompt_embeds_list[1][0][:,:cut_to]], dim=-1)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device,dtype=unet_dtype, )
    
    
    bs_embed, seq_len, _ = prompt_embeds.shape
    
    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        
        
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=unet_dtype, device=device)
        
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
    

    return prompt_embeds, negative_prompt_embeds,pooled_prompt_embeds, negative_pooled_prompt_embeds, class_tokens_mask


def photomaker_clip_v2(clip,model_,prompt_list,negative_prompt,input_id_images,id_embeds,trigger_word="img",num_images_per_prompt=1,nc_flag=False,prompt_2=None,prompt_embeds=None,pooled_prompt_embeds=None,class_tokens_mask=None,prompt_embeds_text_only=None,negative_pooled_prompt_embeds=None):
    id_encoder=model_.id_encoder
    id_image_processor=model_.id_image_processor

    tokenizer=model_.tokenizer_2
    batch_size=len(prompt_list)

    num_id_images = len(input_id_images) #双角色时，须确保输入的图片是列表，且跟prompt对应上
    prompt_arr = prompt_list
    negative_prompt_embeds_arr = []
    prompt_embeds_text_only_arr = []
    prompt_embeds_arr = []
    latents_arr = []
    add_time_ids_arr = []
    negative_pooled_prompt_embeds_arr = []
    pooled_prompt_embeds_text_only_arr = []
    pooled_prompt_embeds_arr = []
    for prompt in prompt_arr:
        (
            prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
            class_tokens_mask
        ) = encode_prompt_with_trigger_word_v2(clip,model_,
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_id_images=num_id_images,
            class_tokens_mask=class_tokens_mask,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            nc_flag=nc_flag,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        # encode, remove trigger word token, then decode
        tokens_text_only = tokenizer.encode(prompt, add_special_tokens=False)
        trigger_word_token = tokenizer.convert_tokens_to_ids(trigger_word)
        if not nc_flag:
            tokens_text_only.remove(trigger_word_token)
        prompt_text_only = tokenizer.decode(tokens_text_only, add_special_tokens=False)
        #print(prompt_text_only)
        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            pooled_prompt_embeds_text_only, # TODO: replace the pooled_prompt_embeds with text only prompt
            negative_pooled_prompt_embeds,
        ) = cf_clip_single(clip,
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 5. Prepare the input ID images
        dtype = next(id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype) # TODO: multiple prompts

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        if not nc_flag:
            if id_embeds is not None:
                id_embeds = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
               
                prompt_embeds = id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds) #torch.Size([1, 1, 3, 224, 224]) torch.Size([1, 77, 2048]) torch.Size([1, 77]) torch.Size([1, 512])
            else:
                prompt_embeds = id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
            # 6. Get the update text embedding with the stacked ID embedding
           
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            pooled_prompt_embeds_arr.append(pooled_prompt_embeds)
            pooled_prompt_embeds = None

        negative_prompt_embeds_arr.append(negative_prompt_embeds)
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds_arr.append(negative_pooled_prompt_embeds)
        negative_pooled_prompt_embeds = None
        prompt_embeds_text_only_arr.append(prompt_embeds_text_only)
        prompt_embeds_text_only = None
        prompt_embeds_arr.append(prompt_embeds)
        prompt_embeds = None
        pooled_prompt_embeds_text_only_arr.append(pooled_prompt_embeds_text_only)
        pooled_prompt_embeds_text_only = None

    emb_dict={"negative_prompt_embeds_arr":negative_prompt_embeds_arr,
              "negative_prompt_embeds":negative_prompt_embeds,
              "negative_pooled_prompt_embeds_arr":negative_pooled_prompt_embeds_arr,
              "negative_pooled_prompt_embeds":negative_pooled_prompt_embeds,
              "prompt_embeds_text_only_arr":prompt_embeds_text_only_arr,
              "prompt_embeds_text_only":prompt_embeds_text_only,
              "prompt_embeds_arr":prompt_embeds_arr,
              "prompt_embeds":prompt_embeds,
              "pooled_prompt_embeds":pooled_prompt_embeds,
              "pooled_prompt_embeds_text_only_arr":pooled_prompt_embeds_text_only_arr,
              "pooled_prompt_embeds_text_only":pooled_prompt_embeds_text_only,
              "pooled_prompt_embeds_arr":pooled_prompt_embeds_arr,
              "batch_size":batch_size,
              }    
    return emb_dict

def Loader_UNO(extra_info,offload,quantize_mode,save_quantezed,lora_rank):
    from .UNO.uno.flux.pipeline import UNOPipeline
    from accelerate import Accelerator
    accelerator = Accelerator()
    device= accelerator.device
    if extra_info is not None:
        model_path = extra_info["unet_path"]
        lora_list=[i for i in [extra_info.get("lora1_path"),extra_info.get("lora2_path")] if i is not None]
        if lora_list:
            lora_ckpt_path=lora_list[0]
            if "dit_lora" not in lora_ckpt_path.lower():
                raise ValueError("lora_ckpt_path must be a dit_lora checkpoint")
            only_lora=True
        else:
            only_lora=False
    
        
        use_fp8=True if quantize_mode=="fp8"  else False

        if  "schnell" in model_path:
            model_type="flux-schnell"
        else:
            model_type="flux-dev"
     
        
        if "schnell" in model_path:
            model_type="flux-schnell"
        else:
            model_type="flux-dev"
        
        if only_lora :
            from .UNO.uno.flux.util import load_flow_model_only_lora
            model_ = load_flow_model_only_lora(
                model_type,model_path,lora_ckpt_path,use_fp8, device="cpu" if offload else device,save_quantezed=save_quantezed, lora_rank=lora_rank)
        else:
            from .UNO.uno.flux.util import load_flow_model
            model_ = load_flow_model(model_type,model_path,use_fp8, device="cpu" if offload else device,save_quantezed=save_quantezed)
        model =  UNOPipeline(
            model_,
            device,
            offload,
            use_fp8,
        )
    else:
        raise ValueError("must use EasyFunction_Lite node")
    return model

def load_pipeline_realcustom(cf_model,realcustom_checkpoint):
    from .RealCustom.models.unet_2d_condition_custom import UNet2DConditionModel as UNet2DConditionModelDiffusers
    if "shallow" in realcustom_checkpoint.lower():
        unet_config_path=os.path.join(cur_path, "RealCustom/configs/realcustom_sigdino_highres_shallow.json")
    else:
        unet_config_path=os.path.join(cur_path, "RealCustom/configs/realcustom_sigdino_highres.json")
    with open(unet_config_path, 'r') as f:
        unet_config = json.load(f)
    
    # Settings for image encoder
    vision_model_config = unet_config.pop("vision_model_config", None)
    vision_model_config_ar = vision_model_config.pop("vision_model_config", None)



    unet_type = unet_config.pop("type", None)
    unet_model = UNet2DConditionModelDiffusers(**unet_config)
    unet_model.eval()

    #cf_state_dict=torch.load(unet_checkpoint,weights_only=False, map_location="cpu")
    cf_state_dict = cf_model.model.diffusion_model.state_dict()
    unet_state_dict = cf_model.model.model_config.process_unet_state_dict_for_saving(cf_state_dict)

    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
    cf_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, unet_model.config)
    unet_model.load_state_dict(cf_state_dict, strict=False)

    realcustom_DICT=torch.load(realcustom_checkpoint,weights_only=False, map_location="cpu")
    unet_model.load_state_dict(realcustom_DICT, strict=False)
    unet_model = torch.compile(unet_model, disable=True)# CHECK THIS

    del cf_state_dict,realcustom_DICT
    gc.collect()
    torch.cuda.empty_cache()
    print("loading unet base model finished.")

    return unet_model,vision_model_config_ar,unet_type




def load_clip_clipvsion(clip_paths,token_paths,dino_path,siglip_path,vision_model_config):
    from .RealCustom.models.text import TextModel
    from transformers import CLIPTextModel,CLIPTextConfig
    from .RealCustom.utils import instantiate_from_config
    
    # text_encoder=CLIPTextModel.from_pretrained(clip_paths[0], local_files_only=True)
    # text_encoder_2=CLIPTextModel.from_pretrained(clip_paths[1], local_files_only=True)

    from safetensors.torch import load_file
    text_encoder_config=CLIPTextConfig.from_pretrained(token_paths[0], local_files_only=True)
    text_encoder = CLIPTextModel(text_encoder_config)
    text_encoder_sd_1 = load_file(clip_paths[0])
    text_encoder.load_state_dict(text_encoder_sd_1, strict=False)
    
    
    text_encoder_config_2=CLIPTextConfig.from_pretrained(token_paths[1], local_files_only=True)
    text_encoder_2 = CLIPTextModel(text_encoder_config_2)
    text_encoder_sd_2 = load_file(clip_paths[1])
    text_encoder_2.load_state_dict(text_encoder_sd_2, strict=False)
    del text_encoder_sd_2,text_encoder_sd_1
    gc.collect()


    text_model = TextModel([text_encoder,text_encoder_2],token_paths,["penultimate_nonorm"])
    text_model.eval()

    siglip_path = os.path.normpath(siglip_path).replace("\\", "/")
    dino_path = os.path.normpath(dino_path).replace("\\", "/")

    vision_model_config["params"]["siglip_path"] = siglip_path
    vision_model_config["params"]["dino_path"] = dino_path

    vision_model = instantiate_from_config(vision_model_config)
    vision_model = vision_model.eval().to(device)

    print("loading image model and text_model finished.")
    return text_model,vision_model




def realcustom_clip_emb(text_model,vision_model,vae_config,vae_downsample_factor,positive_prompt,negative_prompt,positive_image,target_phrase,width,height,device,samples_per_prompt,guidance_weight=7.5):
    from  .RealCustom.inference.inference_utils import find_phrase_positions_in_text
    import torchvision
    with torch.no_grad():
        image_metadata_validate = torch.tensor(
            data=[
                    width,     # original_height
                    height,    # original_width
                    0,              # coordinate top
                    0,              # coordinate left
                    width,     # target_height
                    height,    # target_width
                ],
                device=device,
                dtype=torch.float32
            ).view(1, -1).repeat(samples_per_prompt, 1)
       
        if guidance_weight != 1:
            text_negative_output = text_model(negative_prompt)
        # Compute target phrases
        target_token = torch.zeros(1, 77).to(device)
        
        positions = find_phrase_positions_in_text(positive_prompt, target_phrase)
        for position in positions:
            prompt_before = positive_prompt[:position] # NOTE We do not need -1 here because the SDXL text encoder does not encode the trailing space.
            prompt_include = positive_prompt[:position+len(target_phrase)]
            #print("prompt before: ", prompt_before, ", prompt_include: ", prompt_include)
            prompt_before_length = text_model.get_vaild_token_length(prompt_before) + 1
            prompt_include_length = text_model.get_vaild_token_length(prompt_include) + 1
           # print("prompt_before_length: ", prompt_before_length, ", prompt_include_length: ", prompt_include_length)
            target_token[:, prompt_before_length:prompt_include_length] = 1

        # Text used for progress bar
        pbar_text = positive_prompt[:40]

        # Compute text embeddings
        text_positive_output = text_model(positive_prompt)
        text_positive_embeddings = text_positive_output.embeddings.repeat_interleave(samples_per_prompt, dim=0)
        text_positive_pooled = text_positive_output.pooled[-1].repeat_interleave(samples_per_prompt, dim=0)
        if guidance_weight != 1:
            text_negative_embeddings = text_negative_output.embeddings.repeat_interleave(samples_per_prompt, dim=0)
            text_negative_pooled = text_negative_output.pooled[-1].repeat_interleave(samples_per_prompt, dim=0)
        
        # Compute image embeddings
    # positive_image = Image.open(positive_promt_image_path).convert("RGB")
        positive_image = torchvision.transforms.ToTensor()(positive_image)

        positive_image = positive_image.unsqueeze(0).repeat_interleave(samples_per_prompt, dim=0)
        positive_image = torch.nn.functional.interpolate(
            positive_image, 
            size=(768, 768), 
            mode="bilinear", 
            align_corners=False
        )
        negative_image = torch.zeros_like(positive_image)
        #print("positive_image:",positive_image.size(), negative_image.size()) #torch.Size([1, 3, 768, 768]) torch.Size([1, 3, 768, 768])
        positive_image = positive_image.to(device)
        negative_image = negative_image.to(device)

        positive_image_dict = {"image_ref": positive_image}
        positive_image_output = vision_model(positive_image_dict, device=device)
        #positive_image_output = daul_encoder(vision_model,positive_image_dict)

        negative_image_dict = {"image_ref": negative_image}
        negative_image_output = vision_model(negative_image_dict, device=device)
        #negative_image_output=daul_encoder(vision_model,negative_image_dict)
        # Initialize latent with input latent + noise (i2i) / pure noise (t2i)
        # latent = torch.randn(
        #     size=[
        #         samples_per_prompt,
        #         vae_config["latent_channels"],
        #         height // vae_downsample_factor,
        #         width // vae_downsample_factor
        #     ],
        #     device=device,
        #     generator=torch.Generator(device).manual_seed(seed))
        target_h = (height // vae_downsample_factor) // 2
        target_w = (width // vae_downsample_factor) // 2

        emb_dict={
            "text_positive_embeddings":text_positive_embeddings.to(device),
            "text_positive_pooled":text_positive_pooled.to(device),
            "image_metadata_validate":image_metadata_validate.to(device),
            "positive_image_output":positive_image_output,
            "text_negative_embeddings":text_negative_embeddings.to(device),
            "text_negative_pooled":text_negative_pooled.to(device),
            "negative_image_output":negative_image_output,
        }

        latent_dict={
            "vae_config":vae_config,
            "vae_downsample_factor":vae_downsample_factor,
            "target_h":target_h,
            "target_w":target_w,
            "pbar_text":pbar_text,
            "guidance_weight":guidance_weight,
            "target_token":target_token,
            
        }

        return emb_dict,latent_dict


def realcustom_infer(unet_model,sample_steps,mask_reused_step,emb_dict,latent_dict,mask_scope,mask_strategy,guidance_weight,height,width,seed,
                     device,schedule_type="squared_linear",schedule_shift_snr=1.0,scheduler_type="ddim",unet_prediction="epsilon"):
    from .RealCustom.schedulers.ddim import DDIMScheduler
    from .RealCustom.schedulers.dpm_s import DPMSolverSingleStepScheduler
    from .RealCustom.schedulers.utils import get_betas
    from .RealCustom.inference.mask_generation import mask_generation
    from  .RealCustom.inference.inference_utils import classifier_free_guidance_image_prompt_cascade
    from torchvision.transforms.functional import to_pil_image
    from tqdm import tqdm 
    # Initialize ddim scheduler
    ddim_train_steps = 1000
    ddim_betas = get_betas(name=schedule_type, num_steps=ddim_train_steps, shift_snr=schedule_shift_snr, terminal_pure_noise=False)
    scheduler_class = DPMSolverSingleStepScheduler if scheduler_type == 'dpm' else DDIMScheduler
    scheduler = scheduler_class(betas=ddim_betas, num_train_timesteps=ddim_train_steps, num_inference_timesteps=sample_steps, device=device)
    infer_timesteps = scheduler.timesteps
    #print(infer_timesteps)
    # Real Reverse diffusion process.
    text2image_crossmap_2d_all_timesteps_list = []
    current_step = 0
    pbar_text = latent_dict.get("pbar_text")
    #guidance_weight = latent_dict.get("guidance_weight")
    unet_model.to(device)
    latent=latent_dict.get("latent")
    latent = torch.randn(
            size=[
                1,
                latent_dict.get("vae_config")["latent_channels"],
                height // latent_dict.get("vae_downsample_factor"),
                width // latent_dict.get("vae_downsample_factor")
            ],
            device=device,
            generator=torch.Generator(device).manual_seed(seed))
    

    with torch.no_grad():
        for timestep in tqdm(iterable=infer_timesteps, desc=f"[{pbar_text}]", dynamic_ncols=True):
            
            if current_step < mask_reused_step:
                pred_cond, pred_cond_dict = unet_model(
                    sample=latent,
                    timestep=timestep,
                    encoder_hidden_states=emb_dict.get("text_positive_embeddings"),
                    encoder_attention_mask=None,
                    added_cond_kwargs=dict(
                        text_embeds=emb_dict.get("text_positive_pooled"),
                        time_ids=emb_dict.get("image_metadata_validate")
                    ),
                    vision_input_dict=None,
                    vision_guided_mask=None,
                    return_as_origin=False,
                    return_text2image_mask=True,
                )
                
                crossmap_2d_avg = mask_generation(
                    crossmap_2d_list=pred_cond_dict.get("text2image_crossmap_2d",[]), selfmap_2d_list=pred_cond_dict.get("self_attention_map", []), 
                    target_token=latent_dict.get("target_token"), mask_scope=mask_scope,
                    mask_target_h=latent_dict.get("target_h"), mask_target_w=latent_dict.get("target_w"), mask_mode=mask_strategy,
                )
            else:
                # using previous step's mask
                crossmap_2d_avg = text2image_crossmap_2d_all_timesteps_list[-1].squeeze(1)
            if crossmap_2d_avg.dim() == 5: # Means that each layer uses a separate mask weight.
                text2image_crossmap_2d_all_timesteps_list.append(crossmap_2d_avg.mean(dim=2).unsqueeze(1))
            else:
                text2image_crossmap_2d_all_timesteps_list.append(crossmap_2d_avg.unsqueeze(1))

            pred_cond, pred_cond_dict = unet_model(
                sample=latent,
                timestep=timestep,
                encoder_hidden_states=emb_dict.get("text_positive_embeddings"),
                encoder_attention_mask=None,
                added_cond_kwargs=dict(
                    text_embeds=emb_dict.get("text_positive_pooled"),
                    time_ids=emb_dict.get("image_metadata_validate")
                ),
                vision_input_dict=emb_dict.get("positive_image_output"),
                vision_guided_mask=crossmap_2d_avg,
                return_as_origin=False,
                return_text2image_mask=True,
                multiple_reference_image=False
            )

            crossmap_2d_avg_neg = crossmap_2d_avg.mean(dim=1, keepdim=True)
            pred_negative, pred_negative_dict = unet_model(
                sample=latent,
                timestep=timestep,
                encoder_hidden_states=emb_dict.get("text_negative_embeddings"),
                encoder_attention_mask=None,
                added_cond_kwargs=dict(
                    text_embeds=emb_dict.get("text_negative_pooled"),
                    time_ids=emb_dict.get("image_metadata_validate")
                ),
                vision_input_dict=emb_dict.get("negative_image_output"),
                vision_guided_mask=crossmap_2d_avg,
                return_as_origin=False,
                return_text2image_mask=True,
                multiple_reference_image=False
            )

            pred = classifier_free_guidance_image_prompt_cascade(
                pred_t_cond=None, pred_ti_cond=pred_cond, pred_uncond=pred_negative, 
                guidance_weight_t=guidance_weight, guidance_weight_i=guidance_weight, 
                guidance_stdev_rescale_factor=0, cfg_rescale_mode="naive_global_direct"
            )
            step = scheduler.step(
                model_output=pred,
                model_output_type=unet_prediction,
                timestep=timestep,
                sample=latent,
                )

            latent = step.prev_sample

            current_step += 1

        sample=step.pred_original_sample/0.13025
       
        return sample
    

def load_realcustom_vae(vae,device):
    from .RealCustom.models.vae import AutoencoderKL
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint

    vae_config_path = os.path.join(cur_path, "local_repo/vae/config.json")
    # Initialize vae model
    with open(vae_config_path, 'r') as vae_config_file:
        vae_config = json.load(vae_config_file)
    vae_downsample_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) # 2 ** 3 = 8
    vae_model = AutoencoderKL(**vae_config)
    vae_model.eval().to(device)

    vae_state_dict=vae.get_sd()
    #AE = AutoencoderKL.from_config(ae_config).to(device, torch.float16)
    cf_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, vae_model.config)
    vae_model.load_state_dict(cf_state_dict, strict=False)
    del cf_state_dict,vae_state_dict

   
    #vae_decoder = torch.compile(lambda x: vae_model.decode(x / vae_model.scaling_factor).sample.clip(-1, 1), disable=True)
    vae_encoder = torch.compile(lambda x: vae_model.encode(x).latent_dist.mode().mul_(vae_model.scaling_factor), disable=True)
    print("loading vae finished.")
    return vae_encoder,vae_downsample_factor,vae_config



def load_pipeline_instant_character(extra_info,ip_adapter_path,VAE,quantize_mode):
    from .InstantCharacter.pipeline_wrapper import InstantCharacterFluxPipeline
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig,FluxTransformer2DModel
    if extra_info.get("svdq_repo") is not None:
        print("use svdq")
        from nunchaku import NunchakuFluxTransformer2dModel    
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(extra_info.get("svdq_repo"),offload=True)
        try:
            transformer.set_attention_impl("nunchaku-fp16")
        except:
            pass
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
        pipe = InstantCharacterFluxPipeline.from_pretrained(os.path.join(cur_path,"config/FLUX.1-dev"),vae=vae,transformer=transformer,text_encoder=None,text_encoder_2=None, torch_dtype=torch.bfloat16)
    elif extra_info.get("gguf_path") is not None:
        print("use gguf quantization")   
        from diffusers import  GGUFQuantizationConfig
        transformer = FluxTransformer2DModel.from_single_file(
            extra_info.get("gguf_path") ,
            config=os.path.join(cur_path, "config/FLUX.1-dev/transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
        pipe = InstantCharacterFluxPipeline.from_pretrained(os.path.join(cur_path,"config/FLUX.1-dev"),vae=vae,transformer=transformer,text_encoder=None,text_encoder_2=None, torch_dtype=torch.bfloat16)
    elif extra_info.get("unet_path"):

        if quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_single_file(extra_info.get("unet_path"), config=os.path.join(cur_path,"config/FLUX.1-dev/transformer/config.json"),
                                                                            torch_dtype=torch.bfloat16)
        else:
            from safetensors.torch import load_file
            t_state_dict=load_file(extra_info.get("unet_path"))
            unet_config = FluxTransformer2DModel.load_config(os.path.join(cur_path,"config/FLUX.1-dev/transformer/config.json"))
            transformer = FluxTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
            transformer.load_state_dict(t_state_dict, strict=False)
            del t_state_dict
            gc_cleanup()
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
        pipe = InstantCharacterFluxPipeline.from_pretrained(os.path.join(cur_path,"config/FLUX.1-dev"),transformer=transformer,vae=vae,text_encoder=None,text_encoder_2=None, torch_dtype=torch.bfloat16)
    else:
        print(f"use {quantize_mode} quantization") 
        find_flux=[i for i in [extra_info.get("repo1_path") , extra_info.get("repo2_path")] if "flux" in i ]
        if  not find_flux:
            raise ValueError("can not find flux repo")
        if quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_pretrained(
                find_flux[0],
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True,),
                torch_dtype=torch.bfloat16,
            )
                   
        elif quantize_mode=="nf4": #nf4
            transformer = FluxTransformer2DModel.from_pretrained(
                find_flux[0],
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                ),
                torch_dtype=torch.bfloat16,)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                find_flux[0],
                subfolder="transformer",
                 torch_dtype=torch.bfloat16,
                )

        pipe = InstantCharacterFluxPipeline.from_pretrained(find_flux[0],transformer=transformer,text_encoder=None,text_encoder_2=None, torch_dtype=torch.bfloat16)
    # if not cf_model.get("use_svdq"):
    #     pipe.enable_model_cpu_offload()
    pipe.to(device)
    pipe.init_adapter(
        subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
    )
    
    return pipe

def cf_flux_prompt_clip(cf_clip,prompt):
    tokens = cf_clip.tokenize(prompt)
    tokens["t5xxl"] = cf_clip.tokenize(prompt)["t5xxl"]
    prompt_embeds = cf_clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True).pop("cond")
    tokens["l"] = cf_clip.tokenize(prompt)["l"]
    pooled_prompt_embeds = cf_clip.encode_from_tokens(tokens, return_dict=True).pop("pooled_output")
    prompt_embeds=prompt_embeds.to(device,torch.bfloat16)
    pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.bfloat16)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=torch.bfloat16)
    return prompt_embeds,pooled_prompt_embeds,text_ids


def load_dual_clip(image_encoder_path,image_encoder_2_path,device,dtype):
    from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
    # image encoder
    print(f"=> loading image_encoder_1: {image_encoder_path}")
    image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
    image_processor = SiglipImageProcessor.from_pretrained(image_encoder_path)
    image_encoder.eval()
    image_encoder.to(device, dtype=dtype)


    # image encoder 2
    print(f"=> loading image_encoder_2: {image_encoder_2_path}")
    image_encoder_2 = AutoModel.from_pretrained(image_encoder_2_path)
    image_processor_2 = AutoImageProcessor.from_pretrained(image_encoder_2_path)
    image_encoder_2.eval()
    image_encoder_2.to(device, dtype=dtype)
    image_processor_2.crop_size = dict(height=384, width=384)
    image_processor_2.size = dict(shortest_edge=384)


    return image_encoder,image_processor,image_encoder_2,image_processor_2



@torch.inference_mode()
def instant_character_id_clip(subject_image,siglip_image_encoder,siglip_image_processor,dino_image_encoder_2,dino_image_processor_2,device,dtype):

   
    from einops import rearrange
    
    def encode_siglip_image_emb(siglip_image, device, dtype):
        siglip_image = siglip_image.to(device, dtype=dtype)
        res = siglip_image_encoder(siglip_image, output_hidden_states=True)

        siglip_image_embeds = res.last_hidden_state

        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        
        return siglip_image_embeds, siglip_image_shallow_embeds


    def encode_dinov2_image_emb(dinov2_image, device, dtype):
        dinov2_image = dinov2_image.to(device, dtype=dtype)
        res = dino_image_encoder_2(dinov2_image, output_hidden_states=True)

        dinov2_image_embeds = res.last_hidden_state[:, 1:]

        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)

        return dinov2_image_embeds, dinov2_image_shallow_embeds
    


    def encode_image_emb(siglip_image, device, dtype):
        object_image_pil = siglip_image
        object_image_pil_low_res = [object_image_pil.resize((384, 384))]
        object_image_pil_high_res = object_image_pil.resize((768, 768))
        object_image_pil_high_res = [
            object_image_pil_high_res.crop((0, 0, 384, 384)),
            object_image_pil_high_res.crop((384, 0, 768, 384)),
            object_image_pil_high_res.crop((0, 384, 384, 768)),
            object_image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res)

        siglip_image_embeds = encode_siglip_image_emb(
            siglip_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )
        dinov2_image_embeds = encode_dinov2_image_emb(
            dino_image_processor_2(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )

        image_embeds_low_res_deep = torch.cat([siglip_image_embeds[0], dinov2_image_embeds[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_image_embeds[1], dinov2_image_embeds[1]], dim=2)

        siglip_image_high_res = siglip_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        siglip_image_high_res = siglip_image_high_res[None]
        siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')
        siglip_image_high_res_embeds = encode_siglip_image_emb(siglip_image_high_res, device, dtype)
        siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        dinov2_image_high_res = dino_image_processor_2(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        dinov2_image_high_res = dinov2_image_high_res[None]
        dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
        dinov2_image_high_res_embeds = encode_dinov2_image_emb(dinov2_image_high_res, device, dtype)
        dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow.to(device),
            image_embeds_low_res_deep=image_embeds_low_res_deep.to(device),
            image_embeds_high_res_deep=image_embeds_high_res_deep.to(device),
        )

        return image_embeds_dict

    subject_image = subject_image.resize((max(subject_image.size), max(subject_image.size)))
    subject_image_embeds_dict = encode_image_emb(subject_image, device, dtype)

    return subject_image_embeds_dict

def Loader_Dreamo(extra_info,VAE,quantize_mode,dreamo_lora_path,cfg_distill_path,Turbo_path,device,dreamo_version):
    from .DreamO.dreamo.dreamo_pipeline import DreamOPipeline
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig,FluxTransformer2DModel
    #vae=convert_cfvae2diffuser(VAE,use_flux=True)
    repo_list=[i for i  in [extra_info.get("repo1_path"),extra_info.get("repo2_path")] if i is not None]
    if repo_list:
        flux_repo=[i for i in repo_list if "flux" in i.lower()]
    else:
        raise "you must fill a flux repo"
    if extra_info.get("gguf_path") is not None: #get error
        print("use gguf quantization")
        from diffusers import  GGUFQuantizationConfig
        transformer = FluxTransformer2DModel.from_single_file(
            extra_info.get("gguf_path"),
            config=os.path.join(cur_path, "config/FLUX.1-dev/transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        dreamo_pipeline = DreamOPipeline.from_pretrained(flux_repo, transformer=transformer,torch_dtype=torch.bfloat16)
        dreamo_pipeline.load_dreamo_model(device,dreamo_lora_path,cfg_distill_path,Turbo_path, use_turbo=True,dreamo_version=dreamo_version)
    elif extra_info.get("svdq_repo") is not None:#get error
        print("use svdq")
        from nunchaku import NunchakuFluxTransformer2dModel    
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(extra_info.get("svdq_repo"),offload=True)
        try:
            transformer.set_attention_impl("nunchaku-fp16")
        except:
            pass
        dreamo_pipeline = DreamOPipeline.from_pretrained(flux_repo, transformer=transformer,torch_dtype=torch.bfloat16)
        dreamo_pipeline.load_dreamo_model(device,dreamo_lora_path,cfg_distill_path,Turbo_path, use_turbo=True,use_svdq=True,dreamo_version=dreamo_version)
    elif extra_info.get("unet_path") is not None:
        print("use single unet")
        if quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_single_file(extra_info.get("unet_path"), config=os.path.join(cur_path,"config/FLUX.1-dev/transformer/config.json"),
                                                                            torch_dtype=torch.bfloat16)
        else:
            from safetensors.torch import load_file
            t_state_dict=load_file(extra_info.get("unet_path"))
            unet_config = FluxTransformer2DModel.load_config(os.path.join(cur_path,"config/FLUX.1-dev/transformer/config.json"))
            transformer = FluxTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
            transformer.load_state_dict(t_state_dict, strict=False)
            del t_state_dict
            gc_cleanup()
        dreamo_pipeline = DreamOPipeline.from_pretrained(flux_repo, transformer=transformer,torch_dtype=torch.bfloat16)
        dreamo_pipeline.load_dreamo_model(device,dreamo_lora_path,cfg_distill_path,Turbo_path, use_turbo=True,dreamo_version=dreamo_version)
    else:
        print(f"use {quantize_mode} quantization") 
        if quantize_mode=="fp8":
            # transformer = FluxTransformer2DModel.from_pretrained(
            #     cf_model.get("extra_repo"),
            #     subfolder="transformer",
            #     quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True,),
            #     torch_dtype=torch.bfloat16,
            # )
            dreamo_pipeline = DreamOPipeline.from_pretrained(flux_repo, torch_dtype=torch.bfloat16)
            dreamo_pipeline.load_dreamo_model(device,dreamo_lora_path,cfg_distill_path,Turbo_path, use_turbo=True,dreamo_version=dreamo_version)
            from optimum.quanto import freeze, qint8, quantize
            quantize(dreamo_pipeline.transformer, qint8)
            freeze(dreamo_pipeline.transformer)
            quantize(dreamo_pipeline.text_encoder_2, qint8)
            freeze(dreamo_pipeline.text_encoder_2)  
        
        else:    
            if quantize_mode=="nf4": #nf4
                transformer = FluxTransformer2DModel.from_pretrained(
                    flux_repo,
                    subfolder="transformer",
                    quantization_config=DiffusersBitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                    ),
                    torch_dtype=torch.bfloat16,)
            else:
                transformer = FluxTransformer2DModel.from_pretrained(
                   flux_repo,
                    subfolder="transformer",
                        torch_dtype=torch.bfloat16,
                    )
            dreamo_pipeline = DreamOPipeline.from_pretrained(flux_repo, transformer=transformer,torch_dtype=torch.bfloat16)
            dreamo_pipeline.load_dreamo_model(device,dreamo_lora_path,cfg_distill_path,Turbo_path, use_turbo=True,dreamo_version=dreamo_version)
    dreamo_pipeline.enable_model_cpu_offload()
    return dreamo_pipeline

@torch.no_grad()
def get_align_face(img,face_helper):
    from .DreamO.dreamo.utils import img2tensor,tensor2img
    from torchvision.transforms.functional import normalize
    # the face preprocessing code is same as PuLID
    face_helper.clean_all()
    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face_helper.read_image(image_bgr)
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        return None
    align_face = face_helper.cropped_faces[0]

    input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
    input = input.to(torch.device("cuda"))
    parsing_out = face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
    parsing_out = parsing_out.argmax(dim=1, keepdim=True)
    bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
    bg = sum(parsing_out == i for i in bg_label).bool()
    white_image = torch.ones_like(input)
    # only keep the face features
    face_features_image = torch.where(bg, white_image, input)
    face_features_image = tensor2img(face_features_image, rgb2bgr=False)

    return face_features_image

def Dreamo_image_encoder(BEN2_path,ref_image1,ref_image2,ref_task1,ref_task2,ref_res):
    from .DreamO.dreamo.utils import img2tensor, resize_numpy_image_area
    ref_conds = []
    #debug_images = []

    ref_images = [ref_image1, ref_image2]
    ref_tasks = [ref_task1, ref_task2]

    for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
        if ref_image is not None:
            if ref_task == "id":
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                face_helper = FaceRestoreHelper(
                    upscale_factor=1,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model='retinaface_resnet50',
                    save_ext='png',
                    device=device,
                )
                ref_image = resize_numpy_image_long(ref_image, 1024)
                ref_image = get_align_face(ref_image,face_helper)
            elif ref_task != "style": # ip
                from .DreamO.tools import BEN2
                bg_rm_model = BEN2.BEN_Base().to(device).eval() #TODO 反复加载 需要修复
                #hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
                bg_rm_model.loadcheckpoints(BEN2_path)
                ref_image = bg_rm_model.inference(ref_image) #NEED TO CHECK
                bg_rm_model.to(torch.device('cpu'))
            if ref_task != "id":
                ref_image = resize_numpy_image_area(np.array(ref_image), ref_res * ref_res)
                
            #debug_images.append(ref_image)
            ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
            ref_image = 2 * ref_image - 1.0
            ref_conds.append(
                {
                    'img': ref_image,
                    'task': ref_task,
                    'idx': idx + 1,
                }
            )
    return ref_conds

def Loader_Flux_Diffuser(extra_info,omi_lora_path,VAE,quantize_mode):
    from .OmniConsistency.src_inference.pipeline import FluxPipeline as FluxPipeline_dif
    from .OmniConsistency.src_inference.pipeline_ import FluxPipeline as FluxPipeline_dif_original
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig,FluxTransformer2DModel
    from .OmniConsistency.src_inference.lora_helper import set_single_lora
    flux_dif_repo=os.path.join(cur_path,"config/FLUX.1-dev")
    if extra_info.get("gguf_path") is not None: 
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
        print("use gguf quantization")
     
        from diffusers import  GGUFQuantizationConfig
        transformer = FluxTransformer2DModel.from_single_file(
            extra_info.get("gguf_path"),
            config=os.path.join(cur_path, "config/FLUX.1-dev/transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxPipeline_dif.from_pretrained(flux_dif_repo, vae=vae,transformer=transformer,text_encoder=None,text_encoder_2=None,torch_dtype=torch.bfloat16)
       
    elif extra_info.get("svdq_repo") is not None:
        print("use svdq")
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
        from nunchaku import NunchakuFluxTransformer2dModel    
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(extra_info.get("svdq_repo"),offload=True)
        try:
            transformer.set_attention_impl("nunchaku-fp16")
        except:
            pass
        pipeline = FluxPipeline_dif.from_pretrained(flux_dif_repo, vae=vae,transformer=transformer,text_encoder=None,text_encoder_2=None,torch_dtype=torch.bfloat16)
       
    elif extra_info.get("unet_path") is not None:
        print("use single unet")
        vae=convert_cfvae2diffuser(VAE,use_flux=True)
        if quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_single_file(extra_info.get("unet_path"), config=os.path.join(cur_path,"config/FLUX.1-dev/transformer/config.json"),
                                                                            torch_dtype=torch.bfloat16)
        else:
            from safetensors.torch import load_file
            t_state_dict=load_file(extra_info.get("unet_path"))
            unet_config = FluxTransformer2DModel.load_config(os.path.join(cur_path,"config/FLUX.1-dev/transformer/config.json"))
            transformer = FluxTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
            transformer.load_state_dict(t_state_dict, strict=False)
            del t_state_dict
            gc_cleanup()
        pipeline = FluxPipeline_dif.from_pretrained(flux_dif_repo, vae=vae,transformer=transformer,text_encoder=None,text_encoder_2=None,torch_dtype=torch.bfloat16)
        
    else:
        print(f"use {quantize_mode} quantization") 
        repo_list=[i for i  in [extra_info.get("repo1_path"),extra_info.get("repo2_path")] if i is not None]
        if repo_list:
            flux_repo=[i for i in repo_list if "flux" in i.lower()]
        else:
            raise "you must fill a flux repo"
        if quantize_mode=="nf4": #nf4
            transformer = FluxTransformer2DModel.from_pretrained(
                flux_repo[0],
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                ),
                torch_dtype=torch.bfloat16,)
        elif quantize_mode=="fp8":
            transformer = FluxTransformer2DModel.from_pretrained(
                flux_repo[0],
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True,),
                torch_dtype=torch.bfloat16,
            )
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                flux_repo[0],
                subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                )
        pipeline = FluxPipeline_dif_original.from_pretrained(flux_repo[0],transformer=transformer,torch_dtype=torch.bfloat16)
    
    multi_lora=[]
    if extra_info.get("lora1_path") is not None:
        multi_lora.append(extra_info.get("lora1_path"))
    if extra_info.get("lora2_path") is not None:
        multi_lora.append(extra_info.get("lora2_path"))
    if multi_lora:
        if len(multi_lora)==1:
            set_single_lora(pipeline.transformer, omi_lora_path, lora_weights=[1], cond_size=512)
            pipeline.unload_lora_weights() 
            pipeline.load_lora_weights(os.path.dirname(multi_lora[0]),weight_name=os.path.basename(multi_lora[0]))      
            
        else:
            from .OmniConsistency.src_inference.lora_helper  import set_multi_lora
            lora_weights = [1]*len(multi_lora)
            set_multi_lora(pipeline.transformer, multi_lora, lora_weights=[[lora_weights]], cond_size=512)
            pipeline.unload_lora_weights() 
    else:
        raise "need chocie a  lora model in EasyFunction_Lite!"
    pipeline.enable_model_cpu_offload()
    return pipeline



def load_lora_for_unet_only(pipeline, lora_path, adapter_name="default", lora_scale=1.0):

    try:
      
        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
        pipeline.set_adapters(adapter_name, adapter_weights=lora_scale)
        print(f"成功加载LoRA权重: {adapter_name} (scale: {lora_scale})")
    except Exception as e:
        print(f"加载LoRA权重失败: {e}")
    
    return pipeline
