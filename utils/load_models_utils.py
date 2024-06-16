import yaml
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from .pipeline import PhotoMakerStableDiffusionXLPipeline
import os
import sys
import diffusers
dif_version = str(diffusers.__version__)
dif_version_int= int(dif_version.split(".")[1])
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)
yaml_path = os.path.join(path_dir ,'config','models.yaml')
original_config_file=os.path.join(path_dir,'config','sd_xl_base.yaml')
def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = path.replace('\\', "/")
    return instance_path

yaml_path=get_instance_path(yaml_path)
original_config_file = get_instance_path(original_config_file)
#print(yaml_path,original_config_file)
def get_models_dict():
    # 打开并读取YAML文件
    with open(yaml_path, 'r') as stream:
        try:
            # 解析YAML文件内容
            data = yaml.safe_load(stream)
            
            # 此时 'data' 是一个Python字典，里面包含了YAML文件的所有数据
            # print(data)
            return data
            
        except yaml.YAMLError as exc:
            # 如果在解析过程中发生了错误，打印异常信息
            print(exc)

def load_models(model_info,_sd_type,device,photomaker_path,lora,trigger_words):
    path =  model_info["path"]
    single_files =  model_info["single_files"]
    use_safetensors = model_info["use_safetensors"]
    model_type = model_info["model_type"]

    if model_type == "original":
        if single_files:
            if dif_version_int>=28:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    pretrained_model_link_or_path=path, original_config=original_config_file, torch_dtype=torch.float16)

            else:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    pretrained_model_link_or_path=path, original_config_file=original_config_file, torch_dtype=torch.float16
                )

        else:
            if _sd_type=="Playground_v2p5":
                pipe = DiffusionPipeline.from_pretrained(
                    path, torch_dtype=torch.float16, use_safetensors=use_safetensors
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    path, torch_dtype=torch.float16,use_safetensors=use_safetensors
                )

        pipe = pipe.to(device)
        if lora != "none":
            pipe.load_lora_weights(lora, adapter_name=trigger_words)
            pipe.set_adapters(trigger_words)
            pipe.fuse_lora()
            # pipe._lora_scale=lora_scale

    elif model_type == "Photomaker":
        if single_files:
            #print("loading from a single_files")
            if dif_version_int>=28:
                pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                    pretrained_model_link_or_path=path, original_config=original_config_file, torch_dtype=torch.float16,use_safetensors=use_safetensors)
            else:
                pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                    pretrained_model_link_or_path=path, original_config_file=original_config_file, torch_dtype=torch.float16,use_safetensors=use_safetensors
                )
        else:
            if _sd_type=="Playground_v2p5":
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    path, torch_dtype=torch.float16, use_safetensors=use_safetensors
                )
            else:
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    path, torch_dtype=torch.float16,use_safetensors=use_safetensors
                )

        pipe = pipe.to(device)
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"  # define the trigger word
        )
        if lora != "none":
            pipe.load_lora_weights(lora, adapter_name=trigger_words)
            pipe.set_adapters(trigger_words)
            # pipe._lora_scale=lora_scale
        pipe.fuse_lora()
    else:
        raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {model_type}")
    return pipe

