# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
import json
import os
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image

from ..models.text import TextModel
from ..models.vae import AutoencoderKL

from ..models.unet_2d_condition_custom import UNet2DConditionModel as UNet2DConditionModelDiffusers
from ..schedulers.ddim import DDIMScheduler
from ..schedulers.dpm_s import DPMSolverSingleStepScheduler
from ..schedulers.utils import get_betas

from .inference_utils import find_phrase_positions_in_text, classifier_free_guidance_image_prompt_cascade
from .mask_generation import mask_generation
from ..utils import instantiate_from_config

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)

parser.add_argument("--samples_per_prompt", type=int, required=True)
parser.add_argument("--nrow", type=int, default=4)
parser.add_argument("--sample_steps", type=int, required=True)
parser.add_argument("--schedule_type", type=str, default="squared_linear")                                            # default, `squared_linear
parser.add_argument("--scheduler_type", type=str, default="dpm", choices=["ddim", "dpm"])                             # default, "dpm"
parser.add_argument("--schedule_shift_snr", type=float, default=1)                                                    # default, 1

parser.add_argument("--text_encoder_variant", type=str, nargs="+")
parser.add_argument("--vae_config", type=str, default="configs/vae.json")                                             # default
parser.add_argument("--vae_checkpoint", type=str, required=True)
parser.add_argument("--unet_config", type=str, required=True)
parser.add_argument("--unet_checkpoint", type=str, required=True)
parser.add_argument("--unet_checkpoint_base_model", type=str, default="")
parser.add_argument("--unet_prediction", type=str, choices=DDIMScheduler.prediction_types, default="epsilon")          # default, "epsilon"

parser.add_argument("--negative_prompt", type=str, default="prompts/validation_negative.txt")                          # default

parser.add_argument("--compile", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, required=True)

parser.add_argument("--guidance_weight", type=float, default=7.5)
parser.add_argument("--seed", type=int, default=666)
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--text_prompt", type=str, required=True)
parser.add_argument("--image_prompt_path", type=str, required=True)
parser.add_argument("--target_phrase", type=str, required=True)
parser.add_argument("--mask_scope", type=float, default=0.20)
parser.add_argument("--mask_strategy",  type=str, nargs="+", default=["max_norm"])
parser.add_argument("--mask_reused_step", type=int, default=12)

args = parser.parse_args()

# Initialize unet model
with open(args.unet_config) as unet_config_file:
    unet_config = json.load(unet_config_file)

    # Settings for image encoder
    vision_model_config = unet_config.pop("vision_model_config", None)
    args.vision_model_config = vision_model_config.pop("vision_model_config", None)

    unet_type = unet_config.pop("type", None)
    unet_model = UNet2DConditionModelDiffusers(**unet_config)

unet_model.eval().to(args.device)
unet_model.load_state_dict(torch.load(args.unet_checkpoint, map_location=args.device), strict=False)
print("loading unet model finished.")

if args.unet_checkpoint_base_model != "":
    if "safetensors" in args.unet_checkpoint_base_model:
        from safetensors import safe_open
        tensors = {}
        with safe_open(args.unet_checkpoint_base_model, framework="pt", device='cpu') as f:
            for k in f.keys():
                new_k = k.replace("model.diffusion_model.", "")
                tensors[k] = f.get_tensor(k)
        unet_model.load_state_dict(tensors, strict=False)
    else:
        unet_model.load_state_dict(torch.load(args.unet_checkpoint_base_model, map_location=args.device), strict=False)
unet_model = torch.compile(unet_model, disable=not args.compile)
print("loading unet base model finished.")

# Initialize vae model
with open(args.vae_config) as vae_config_file:
    vae_config = json.load(vae_config_file)
vae_downsample_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) # 2 ** 3 = 8
vae_model = AutoencoderKL(**vae_config)
vae_model.eval().to(args.device)
vae_model.load_state_dict(torch.load(args.vae_checkpoint, map_location=args.device))
vae_decoder = torch.compile(lambda x: vae_model.decode(x / vae_model.scaling_factor).sample.clip(-1, 1), disable=not args.compile)
vae_encoder = torch.compile(lambda x: vae_model.encode(x).latent_dist.mode().mul_(vae_model.scaling_factor), disable=not args.compile)
print("loading vae finished.")

# Initialize ddim scheduler
ddim_train_steps = 1000
ddim_betas = get_betas(name=args.schedule_type, num_steps=ddim_train_steps, shift_snr=args.schedule_shift_snr, terminal_pure_noise=False)
scheduler_class = DPMSolverSingleStepScheduler if args.scheduler_type == 'dpm' else DDIMScheduler
scheduler = scheduler_class(betas=ddim_betas, num_train_timesteps=ddim_train_steps, num_inference_timesteps=args.sample_steps, device=args.device)
infer_timesteps = scheduler.timesteps

# Initialize text model
text_model = TextModel(args.text_encoder_variant, ["penultimate_nonorm"])
text_model.eval().to(args.device)
print("loading text model finished.")

# Initialize image model.
vision_model = instantiate_from_config(args.vision_model_config)
vision_model = vision_model.eval().to(args.device)
print("loading image model finished.")

negative_prompt = ""
if args.negative_prompt:
    with open(args.negative_prompt) as f:
        negative_prompt = f.read().strip()

image_metadata_validate = torch.tensor(
    data=[
        args.width,     # original_height
        args.height,    # original_width
        0,              # coordinate top
        0,              # coordinate left
        args.width,     # target_height
        args.height,    # target_width
    ],
    device=args.device,
    dtype=torch.float32
).view(1, -1).repeat(args.samples_per_prompt, 1)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
args.output_image_grid_dir = os.path.join(args.output_dir, "images_grid")
args.output_image_dir = os.path.join(args.output_dir, "images")
args.output_mask_grid_dir = os.path.join(args.output_dir, "masks_grid")
args.output_mask_dir = os.path.join(args.output_dir, "masks")
os.makedirs(args.output_image_grid_dir, exist_ok=True)
os.makedirs(args.output_image_dir, exist_ok=True)
os.makedirs(args.output_mask_grid_dir, exist_ok=True)
os.makedirs(args.output_mask_dir, exist_ok=True)

with torch.no_grad():
    # Prepare negative prompt.
    if args.guidance_weight != 1:
        text_negative_output = text_model(negative_prompt)

    positive_prompt = args.text_prompt
    positive_promt_image_path = args.image_prompt_path
    target_phrase = args.target_phrase

    # Compute target phrases
    target_token = torch.zeros(1, 77).to(args.device)
    positions = find_phrase_positions_in_text(positive_prompt, target_phrase)
    for position in positions:
        prompt_before = positive_prompt[:position] # NOTE We do not need -1 here because the SDXL text encoder does not encode the trailing space.
        prompt_include = positive_prompt[:position+len(target_phrase)]
        print("prompt before: ", prompt_before, ", prompt_include: ", prompt_include)
        prompt_before_length = text_model.get_vaild_token_length(prompt_before) + 1
        prompt_include_length = text_model.get_vaild_token_length(prompt_include) + 1
        print("prompt_before_length: ", prompt_before_length, ", prompt_include_length: ", prompt_include_length)
        target_token[:, prompt_before_length:prompt_include_length] = 1

    # Text used for progress bar
    pbar_text = positive_prompt[:40]

    # Compute text embeddings
    text_positive_output = text_model(positive_prompt)
    text_positive_embeddings = text_positive_output.embeddings.repeat_interleave(args.samples_per_prompt, dim=0)
    text_positive_pooled = text_positive_output.pooled[-1].repeat_interleave(args.samples_per_prompt, dim=0)
    if args.guidance_weight != 1:
        text_negative_embeddings = text_negative_output.embeddings.repeat_interleave(args.samples_per_prompt, dim=0)
        text_negative_pooled = text_negative_output.pooled[-1].repeat_interleave(args.samples_per_prompt, dim=0)
    
    # Compute image embeddings
    positive_image = Image.open(positive_promt_image_path).convert("RGB")
    positive_image = torchvision.transforms.ToTensor()(positive_image)

    positive_image = positive_image.unsqueeze(0).repeat_interleave(args.samples_per_prompt, dim=0)
    positive_image = torch.nn.functional.interpolate(
        positive_image, 
        size=(768, 768), 
        mode="bilinear", 
        align_corners=False
    )
    negative_image = torch.zeros_like(positive_image)
    print(positive_image.size(), negative_image.size())
    positive_image = positive_image.to(args.device)
    negative_image = negative_image.to(args.device)

    positive_image_dict = {"image_ref": positive_image}
    positive_image_output = vision_model(positive_image_dict, device=args.device)

    negative_image_dict = {"image_ref": negative_image}
    negative_image_output = vision_model(negative_image_dict, device=args.device)

    # Initialize latent with input latent + noise (i2i) / pure noise (t2i)
    latent = torch.randn(
        size=[
            args.samples_per_prompt,
            vae_config["latent_channels"],
            args.height // vae_downsample_factor,
            args.width // vae_downsample_factor
        ],
        device=args.device,
        generator=torch.Generator(args.device).manual_seed(args.seed))
    target_h = (args.height // vae_downsample_factor) // 2
    target_w = (args.width // vae_downsample_factor) // 2

    # Real Reverse diffusion process.
    text2image_crossmap_2d_all_timesteps_list = []
    current_step = 0
    for timestep in tqdm(iterable=infer_timesteps, desc=f"[{pbar_text}]", dynamic_ncols=True):
        if current_step < args.mask_reused_step:
            pred_cond, pred_cond_dict = unet_model(
                sample=latent,
                timestep=timestep,
                encoder_hidden_states=text_positive_embeddings,
                encoder_attention_mask=None,
                added_cond_kwargs=dict(
                    text_embeds=text_positive_pooled,
                    time_ids=image_metadata_validate
                ),
                vision_input_dict=None,
                vision_guided_mask=None,
                return_as_origin=False,
                return_text2image_mask=True,
            )
            crossmap_2d_avg = mask_generation(
                crossmap_2d_list=pred_cond_dict["text2image_crossmap_2d"], selfmap_2d_list=pred_cond_dict.get("self_attention_map", []), 
                target_token=target_token, mask_scope=args.mask_scope,
                mask_target_h=target_h, mask_target_w=target_w, mask_mode=args.mask_strategy,
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
            encoder_hidden_states=text_positive_embeddings,
            encoder_attention_mask=None,
            added_cond_kwargs=dict(
                text_embeds=text_positive_pooled,
                time_ids=image_metadata_validate
            ),
            vision_input_dict=positive_image_output,
            vision_guided_mask=crossmap_2d_avg,
            return_as_origin=False,
            return_text2image_mask=True,
            multiple_reference_image=False
        )

        crossmap_2d_avg_neg = crossmap_2d_avg.mean(dim=1, keepdim=True)
        pred_negative, pred_negative_dict = unet_model(
            sample=latent,
            timestep=timestep,
            encoder_hidden_states=text_negative_embeddings,
            encoder_attention_mask=None,
            added_cond_kwargs=dict(
                text_embeds=text_negative_pooled,
                time_ids=image_metadata_validate
            ),
            vision_input_dict=negative_image_output,
            vision_guided_mask=crossmap_2d_avg,
            return_as_origin=False,
            return_text2image_mask=True,
            multiple_reference_image=False
        )

        pred = classifier_free_guidance_image_prompt_cascade(
            pred_t_cond=None, pred_ti_cond=pred_cond, pred_uncond=pred_negative, 
            guidance_weight_t=args.guidance_weight, guidance_weight_i=args.guidance_weight, 
            guidance_stdev_rescale_factor=0, cfg_rescale_mode="naive_global_direct"
        )
        step = scheduler.step(
            model_output=pred,
            model_output_type=args.unet_prediction,
            timestep=timestep,
            sample=latent)

        latent = step.prev_sample

        current_step += 1

    sample = vae_decoder(step.pred_original_sample)

    # save each image 
    for sample_i in range(sample.size(0)):
        sample_i_image = torch.clamp(sample[sample_i] * 0.5 + 0.5, min=0, max=1).float()
        to_pil_image(sample_i_image).save(args.output_image_dir + "/output_{}.jpg".format(sample_i))

    # save grid images
    sample = make_grid(sample, normalize=True, value_range=(-1, 1), nrow=args.nrow).float()
    to_pil_image(sample).save(args.output_image_grid_dir + "/grid_image.jpg")