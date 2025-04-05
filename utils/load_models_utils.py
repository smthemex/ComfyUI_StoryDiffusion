import yaml
import torch
from diffusers import StableDiffusionXLPipeline
from .pipeline import PhotoMakerStableDiffusionXLPipeline

import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = path.replace('\\', "/")
    return instance_path

original_config_file=get_instance_path(os.path.join(path_dir,'config','sd_xl_base.yaml'))
loras_path = get_instance_path(os.path.join(path_dir,"config","lora.yaml"))
add_config=os.path.join(path_dir,"local_repo")

def get_lora_dict():
    # 打开并读取YAML文件
    with open(loras_path, 'r', encoding="UTF-8") as stream:
        try:
            # 解析YAML文件内容
            data = yaml.safe_load(stream)

            # 此时 'data' 是一个Python字典，里面包含了YAML文件的所有数据
            # print(data)
            return data

        except yaml.YAMLError as exc:
            # 如果在解析过程中发生了错误，打印异常信息
            print(exc)

datas = get_lora_dict()
lora_lightning_list = datas["lightning_xl_lora"]

def load_models(path,model_type,single_files,use_safetensors,photo_vesion,photomaker_path,lora,lora_path,trigger_words,lora_scale):
    path=get_instance_path(path)
    if model_type == "txt2img":
        if single_files:
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    path,config=add_config, original_config=original_config_file, torch_dtype=torch.float16)
            except:
                pipe = StableDiffusionXLPipeline.from_single_file(
                        path,config=add_config, original_config_file=original_config_file,
                        torch_dtype=torch.float16
                    )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                path, torch_dtype=torch.float16,use_safetensors=use_safetensors
            )

        if lora:
            if lora in lora_lightning_list:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
            else:
                pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                pipe.fuse_lora(adapter_names=[trigger_words, ], lora_scale=lora_scale)

    elif model_type == "img2img":
        if photo_vesion== "v1" :
            if single_files:
                # print("loading from a single_files")
                try:
                    pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                        path,config=add_config, original_config=original_config_file,
                        torch_dtype=torch.float16, use_safetensors=use_safetensors)
                except:
                    pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                            path, config=add_config,original_config_file=original_config_file,
                            torch_dtype=torch.float16, use_safetensors=use_safetensors
                        )
            
            else:
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    path, torch_dtype=torch.float16, use_safetensors=use_safetensors
                )
            pipe.load_photomaker_adapter(
                photomaker_path,
                subfolder="",
                weight_name="photomaker-v1.bin",
                trigger_word="img"  # define the trigger word
            )
        else:
            from .pipeline_v2 import PhotoMakerStableDiffusionXLPipeline as PhotoMakerStableDiffusionXLPipelineV2
            if single_files:
                # print("loading from a single_files")
                try:
                    pipe = PhotoMakerStableDiffusionXLPipelineV2.from_single_file(
                        path,config=add_config, original_config=original_config_file,
                        torch_dtype=torch.float16, use_safetensors=use_safetensors)
                except:
                    pipe = PhotoMakerStableDiffusionXLPipelineV2.from_single_file(
                            path, config=add_config,original_config_file=original_config_file,
                            torch_dtype=torch.float16, use_safetensors=use_safetensors
                        )
            else:
                pipe = PhotoMakerStableDiffusionXLPipelineV2.from_pretrained(
                    path, torch_dtype=torch.float16, use_safetensors=use_safetensors
                )
            # define the trigger word
            pipe.load_photomaker_adapter(
                "F:/test/ComfyUI/models/photomaker/photomaker-v2.bin",
                subfolder="",
                weight_name="photomaker-v2.bin",
                trigger_word="img",
                pm_version= 'v2',
            )
            
        if lora:
            if lora in lora_lightning_list:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
            else:
                pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                pipe.fuse_lora(adapter_names=[trigger_words, ], lora_scale=lora_scale)

    else:
        print(f"using{model_type}node,must choice{model_type}type in model_loader node")
        raise "load error"
    return pipe

