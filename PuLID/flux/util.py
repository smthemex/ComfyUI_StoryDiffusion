import os
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
import json
from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder
import folder_paths

@dataclass
class SamplingOptions:
    text: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str
    ae_path: str
    repo_id: str
    repo_flow: str
    repo_ae: str


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path='models/flux1-dev.safetensors',
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path='models/ae.safetensors',
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-nf4": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path='models/flux1-dev.safetensors',
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path='models/ae.safetensors',
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

# from XLabs-AI https://github.com/XLabs-AI/x-flux/blob/1f8ef54972105ad9062be69fe6b7f841bce02a08/src/flux/util.py#L330
def load_flow_model_quintized(name: str, ckpt_path,quantized_mode,device: str = "cuda", hf_download: bool = True):
    from optimum.quanto import requantize,qint4,quantize,freeze
    # Loading Flux
    print(f"Init model in {quantized_mode}")
    model = Flux(configs[name].params).to(torch.bfloat16)
    if quantized_mode == "nf4":  #nf4 is now useful.
        json_path = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI_StoryDiffusion", "utils",
                                 "config.json")
        test=False
        if  test:
            from accelerate.utils import set_module_tensor_to_device, compute_module_sizes
            from accelerate import init_empty_weights
            from .convert_nf4_flux import _replace_with_bnb_linear, create_quantized_param, \
                check_quantized_param
            import safetensors.torch
            import gc
            from diffusers import FluxTransformer2DModel
            dtype = torch.bfloat16
            is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
            original_state_dict = safetensors.torch.load_file(ckpt_path)
            print("Start a quantization process...")
            with init_empty_weights():
                config = FluxTransformer2DModel.load_config(json_path)
                model = FluxTransformer2DModel.from_config(config).to(dtype)
                expected_state_dict_keys = list(model.state_dict().keys())
            
            _replace_with_bnb_linear(model, "nf4")
            
            for param_name, param in original_state_dict.items():
                if param_name not in expected_state_dict_keys:
                    continue
                
                is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
                if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
                    param = param.to(dtype)
                
                if not check_quantized_param(model, param_name):
                    set_module_tensor_to_device(model, param_name, device=0, value=param)
                else:
                    create_quantized_param(
                        model, param, param_name, target_device=0, state_dict=original_state_dict,
                        pre_quantized=True
                    )
            
            del original_state_dict
            gc.collect()

        else:
            sd = load_sft(ckpt_path, device='cpu')
            with open(json_path) as f:
                quantization_map = json.load(f)
            
            print("Start a quantization process...")
            requantize(model, sd, quantization_map)
        print("Model is quantized!")
    else: #fp8
        # ckpt_path = 'models/flux-dev-fp8.safetensors'
        if (
                not os.path.exists(ckpt_path)
                and hf_download
        ):
            ckpt_path = hf_hub_download("XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors")
            json_path = hf_hub_download("XLabs-AI/flux-dev-fp8", 'flux_dev_quantization_map.json')
        else:
            json_path = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI_StoryDiffusion", "utils",
                                     "config.json")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device='cpu')
        with open(json_path) as f:
            quantization_map = json.load(f)
        print("Start a quantization process...")
        requantize(model, sd, quantization_map)
        print("Model is quantized!")
    return model

def load_flow_model(name: str,ckpt_path, device: str = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model in fp16")
    if ckpt_path is not None:
        # load_sft doesn't support torch.device
        with torch.device(device):
            model = Flux(configs[name].params).to(torch.bfloat16)
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print_load_warning(missing, unexpected)
    return model


def load_t5(name,clip_cf,quantized_mode,device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    #return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)
    return HFEmbedder(f"{name}/text_encoder_2", f"{name}/tokenizer_2",clip_cf,is_clip=False,quantized_mode=quantized_mode,max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(name,clip_cf,quantized_mode,device= "cuda") -> HFEmbedder:
    return HFEmbedder(f"{name}/text_encoder",f"{name}/tokenizer",clip_cf,is_clip=True,quantized_mode=quantized_mode,max_length=77, torch_dtype=torch.bfloat16).to(device)
    #return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str,vae_cf, device: str = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path=vae_cf
    #ckpt_path = os.path.join(name,"ae.safetensors")   # ckpt_path = configs[name].ae_path

    if (
        not os.path.exists(ckpt_path)
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae, local_dir='models')

    print("Init AE")  # Loading the autoencoder

    with torch.device(device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False)
        print_load_warning(missing, unexpected)
    return ae

