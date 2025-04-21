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

import os
import json
import torch
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
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

from tqdm import tqdm
from einops import rearrange

class RealCustomInferencePipeline:
    def __init__(
        self,
        unet_config,
        unet_checkpoint,
        realcustom_checkpoint,
        vae_config="ckpts/sdxl/vae/sdxl.json",
        vae_checkpoint="ckpts/sdxl/vae/sdxl-vae.pth",
        model_type="bf16", 
        device="cuda",
    ):
        if model_type == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        if not os.path.exists("ckpts/"):
            from huggingface_hub import snapshot_download
            print("Downloading RealCustom ...")
            snapshot_download(
                repo_id="bytedance-research/RealCustom",
                repo_type="model",
                local_dir="ckpts",  # 指定本地目录
                allow_patterns="ckpts/**",  # 只下载 ckpts 文件夹内容
                local_dir_use_symlinks=False,  # 直接存储文件而非符号链接
            )
        
        self.device = device
        self.unet_checkpoint = unet_checkpoint
        self.realcustom_checkpoint = realcustom_checkpoint
        self._load_unet_checkpoint(unet_config, unet_checkpoint, realcustom_checkpoint)
        self._load_vae_checkpoint(vae_config, vae_checkpoint)
        self._load_encoder_checkpoint()
        self._init_scheduler()
        self._load_negative_prompt()

    
    def _load_unet_checkpoint(self, unet_config, unet_checkpoint, realcustom_checkpoint):
        # Initialize unet model
        with open(unet_config) as unet_config_file:
            unet_config = json.load(unet_config_file)
        self.unet_prediction = "epsilon"

        # Settings for image encoder
        vision_model_config = unet_config.pop("vision_model_config", None)
        self.vision_model_config = vision_model_config.pop("vision_model_config", None)

        self.unet_model = UNet2DConditionModelDiffusers(**unet_config)

        self.unet_model.eval().to(self.device).to(self.torch_dtype)
        self.unet_model.load_state_dict(torch.load(unet_checkpoint, map_location=self.device), strict=False)
        self.unet_model.load_state_dict(torch.load(realcustom_checkpoint, map_location=self.device), strict=False)
        print("loading unet model finished.")

    def _reload_unet_checkpoint(self, unet_checkpoint, realcustom_checkpoint):
        self.unet_model.load_state_dict(torch.load(unet_checkpoint, map_location=self.device), strict=False)
        self.unet_model.load_state_dict(torch.load(realcustom_checkpoint, map_location=self.device), strict=False)
        print("reloading unet model finished.")
    
    def _load_vae_checkpoint(self, vae_config, vae_checkpoint):
        # Initialize vae model
        with open(vae_config) as vae_config_file:
            vae_config = json.load(vae_config_file)
        self.latent_channels = vae_config["latent_channels"]
        self.vae_downsample_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) # 2 ** 3 = 8

        vae_model = AutoencoderKL(**vae_config)
        vae_model.eval().to(self.device).to(self.torch_dtype)
        vae_model.load_state_dict(torch.load(vae_checkpoint, map_location=self.device))
        self.vae_decoder = torch.compile(lambda x: vae_model.decode(x / vae_model.scaling_factor).sample.clip(-1, 1), disable=True)
        self.vae_encoder = torch.compile(lambda x: vae_model.encode(x).latent_dist.mode().mul_(vae_model.scaling_factor), disable=True)

        print("loading vae finished.")
    
    def _load_encoder_checkpoint(self, ):
        # Initialize text encoder
        text_encoder_variant = ["ckpts/sdxl/clip-sdxl-1", "ckpts/sdxl/clip-sdxl-2"]
        text_encoder_mode = ["penultimate_nonorm"]
        self.text_model = TextModel(text_encoder_variant, text_encoder_mode)
        self.text_model.eval().to(self.device).to(self.torch_dtype)
        print("loading text model finished.")

        # Initialize image encoder
        self.vision_model = instantiate_from_config(self.vision_model_config)
        self.vision_model.eval().to(self.device).to(self.torch_dtype)
        print("loading image model finished.")
    
    def _init_scheduler(self, ):
        # Initialize ddim scheduler
        ddim_train_steps = 1000
        schedule_type = "squared_linear"
        scheduler_type = "dpm"
        schedule_shift_snr = 1
        self.sample_steps = 25
        ddim_betas = get_betas(name=schedule_type, num_steps=ddim_train_steps, shift_snr=schedule_shift_snr, terminal_pure_noise=False)
        scheduler_class = DPMSolverSingleStepScheduler if scheduler_type == 'dpm' else DDIMScheduler

        self.scheduler = scheduler_class(betas=ddim_betas, num_train_timesteps=ddim_train_steps, num_inference_timesteps=self.sample_steps, device=self.device)
        self.infer_timesteps = self.scheduler.timesteps
    
    def _load_negative_prompt(self, ):
        with open("prompts/validation_negative.txt") as f:
            self.negative_prompt = f.read().strip()
        self.text_negative_output = self.text_model(self.negative_prompt)
    
    def generation(
        self, 
        text, 
        image_pil, 
        target_phrase,

        height=1024,
        width=1024,
        guidance_scale=3.5,
        seed=1234,
        samples_per_prompt=4,

        mask_scope=0.25,

        new_unet_checkpoint="",          # in case you want to change
        new_realcustom_checkpoint="",    # in case you want to change
        mask_strategy=["min_max_per_channel"],
        mask_reused_step=12,
        return_each_image=False,
    ):
        
        if new_unet_checkpoint != "" and new_unet_checkpoint != self.unet_checkpoint:
            self.unet_checkpoint = new_unet_checkpoint
            self.unet_model.load_state_dict(torch.load(new_unet_checkpoint, map_location=self.device), strict=False)
            print("Reloading Unet {} finised.".format(new_unet_checkpoint))
        if new_realcustom_checkpoint != "" and new_realcustom_checkpoint != self.realcustom_checkpoint:
            self.realcustom_checkpoint = new_realcustom_checkpoint
            self.unet_model.load_state_dict(torch.load(new_realcustom_checkpoint, map_location=self.device), strict=False)
            print("Reloading RealCustom {} finised.".format(new_realcustom_checkpoint))

        samples_per_prompt = int(samples_per_prompt)
        image_metadata_validate = self._get_metadata(height, width, samples_per_prompt)
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        seed = int(seed)

        with torch.no_grad(), torch.autocast(self.device, self.torch_dtype):
            target_token = self._find_phrase_positions_in_text(text, target_phrase)

            # Compute text embeddings
            text_positive_output = self.text_model(text)
            text_positive_embeddings = text_positive_output.embeddings.repeat_interleave(samples_per_prompt, dim=0)
            text_positive_pooled = text_positive_output.pooled[-1].repeat_interleave(samples_per_prompt, dim=0)
            if guidance_scale != 1:
                text_negative_embeddings = self.text_negative_output.embeddings.repeat_interleave(samples_per_prompt, dim=0)
                text_negative_pooled = self.text_negative_output.pooled[-1].repeat_interleave(samples_per_prompt, dim=0)

            # Compute image embeddings
            # positive_image = Image.open(image_path).convert("RGB")
            positive_image = image_pil
            positive_image = torchvision.transforms.ToTensor()(positive_image)

            positive_image = positive_image.unsqueeze(0).repeat_interleave(samples_per_prompt, dim=0)
            positive_image = torch.nn.functional.interpolate(
                positive_image, 
                size=(768, 768), 
                mode="bilinear", 
                align_corners=False
            )
            negative_image = torch.zeros_like(positive_image)
            positive_image = positive_image.to(self.device).to(self.torch_dtype)
            negative_image = negative_image.to(self.device).to(self.torch_dtype)

            positive_image_dict = {"image_ref": positive_image}
            positive_image_output = self.vision_model(positive_image_dict, device=self.device)

            negative_image_dict = {"image_ref": negative_image}
            negative_image_output = self.vision_model(negative_image_dict, device=self.device)

            # Initialize latent with input latent
            latent = torch.randn(
                size=[
                    samples_per_prompt,
                    self.latent_channels,
                    height // self.vae_downsample_factor,
                    width // self.vae_downsample_factor
                ],
                device=self.device,
                generator=torch.Generator(self.device).manual_seed(seed)).to(self.torch_dtype)
            target_h = (height // self.vae_downsample_factor) // 2
            target_w = (width // self.vae_downsample_factor) // 2

            text2image_crossmap_2d_all_timesteps_list = []
            current_step = 0
            pbar_text = text[:40]
            for timestep in tqdm(iterable=self.infer_timesteps, desc=f"[{pbar_text}]", dynamic_ncols=True):
                if current_step < mask_reused_step:
                    pred_cond, pred_cond_dict = self.unet_model(
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
                        crossmap_2d_list=pred_cond_dict.get("text2image_crossmap_2d",[]), selfmap_2d_list=pred_cond_dict.get("self_attention_map", []), 
                        target_token=target_token, mask_scope=mask_scope,
                        mask_target_h=target_h, mask_target_w=target_w, mask_mode=mask_strategy,
                    )
                else:
                    # using previous step's mask
                    crossmap_2d_avg = text2image_crossmap_2d_all_timesteps_list[-1].squeeze(1)
                if crossmap_2d_avg.dim() == 5: # Means that each layer uses a separate mask weight.
                    text2image_crossmap_2d_all_timesteps_list.append(crossmap_2d_avg.mean(dim=2).unsqueeze(1))
                else:
                    text2image_crossmap_2d_all_timesteps_list.append(crossmap_2d_avg.unsqueeze(1))

                pred_cond, pred_cond_dict = self.unet_model(
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

                pred_negative, pred_negative_dict = self.unet_model(
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
                    guidance_weight_t=guidance_scale, guidance_weight_i=guidance_scale, 
                    guidance_stdev_rescale_factor=0, cfg_rescale_mode="naive_global_direct"
                )
                step = self.scheduler.step(
                    model_output=pred,
                    model_output_type=self.unet_prediction,
                    timestep=timestep,
                    sample=latent)

                latent = step.prev_sample

                current_step += 1
            sample = self.vae_decoder(step.pred_original_sample)
        
        # save each image
        images_pil_list = []
        for sample_i in range(sample.size(0)):
            sample_i_image = torch.clamp(sample[sample_i] * 0.5 + 0.5, min=0, max=1).float()

            images_pil_list.append(to_pil_image(sample_i_image))
            # to_pil_image(sample_i_image).save("./test_{}.jpg".format(sample_i))

        # save grid images
        sample = make_grid(sample, normalize=True, value_range=(-1, 1), nrow=int(samples_per_prompt ** 0.5)).float()
        # to_pil_image(sample).save("./output_grid_image.jpg")

        # save all masks
        text2image_crossmap_2d_all_timesteps = torch.cat(text2image_crossmap_2d_all_timesteps_list, dim=1)
        text2image_crossmap_2d_all_timesteps = rearrange(text2image_crossmap_2d_all_timesteps, "b t c h w -> (b t) c h w")
        c = text2image_crossmap_2d_all_timesteps.size(1)
        text2image_crossmap_2d_all_timesteps = rearrange(text2image_crossmap_2d_all_timesteps, "B (c 1) h w -> (B c) 1 h w")
        sample_mask = make_grid(text2image_crossmap_2d_all_timesteps, normalize=False, value_range=(-1, 1), nrow=int(self.sample_steps * c))
        # to_pil_image(sample_mask).save("./output_grid_mask.jpg")

        if return_each_image:
            return images_pil_list, to_pil_image(sample), to_pil_image(sample_mask)
        else:
            return to_pil_image(sample), to_pil_image(sample_mask)

    def _get_metadata(self, height, width, samples_per_prompt):
        image_metadata_validate = torch.tensor(
            data=[
                width,     # original_height
                height,    # original_width
                0,         # coordinate top
                0,         # coordinate left
                width,     # target_height
                height,    # target_width
            ],
            device=self.device,
            dtype=self.torch_dtype
        ).view(1, -1).repeat(samples_per_prompt, 1)

        return image_metadata_validate
    
    def _find_phrase_positions_in_text(self, text, target_phrase):
        # Compute target phrases
        target_token = torch.zeros(1, 77).to(self.device)
        positions = find_phrase_positions_in_text(text, target_phrase)
        for position in positions:
            prompt_before = text[:position] # NOTE We do not need -1 here because the SDXL text encoder does not encode the trailing space.
            prompt_include = text[:position+len(target_phrase)]
            print("prompt before: ", prompt_before, ", prompt_include: ", prompt_include)
            prompt_before_length = self.text_model.get_vaild_token_length(prompt_before) + 1
            prompt_include_length = self.text_model.get_vaild_token_length(prompt_include) + 1
            print("prompt_before_length: ", prompt_before_length, ", prompt_include_length: ", prompt_include_length)
            target_token[:, prompt_before_length:prompt_include_length] = 1
        
        return target_token