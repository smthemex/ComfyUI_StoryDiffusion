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

import torch
from dataclasses import dataclass
from abc import ABC
from typing import Optional, Union, List


@dataclass
class SchedulerConversionOutput:
    pred_epsilon: torch.Tensor
    pred_original_sample: torch.Tensor
    pred_velocity: torch.Tensor


@dataclass
class SchedulerStepOutput:
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class Scheduler(ABC):
    prediction_types = ["epsilon", "sample", "v_prediction"]
    timesteps_types = ["leading", "linspace", "trailing"]

    def __init__(
        self,
        num_train_timesteps: int,
        num_inference_timesteps: int,
        betas: torch.Tensor,
        inference_timesteps: Union[str, List[int]] = "trailing",
        set_alpha_to_one: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ):
        assert num_train_timesteps > 0
        assert num_train_timesteps >= num_inference_timesteps
        assert num_train_timesteps == betas.size(0)
        assert betas.ndim == 1

        self.device = device or betas.device
        self.dtype = dtype

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps

        self.betas = betas.to(device=device, dtype=dtype)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0, device=self.device, dtype=self.dtype) if set_alpha_to_one else self.alphas_cumprod[0]

        if isinstance(inference_timesteps, list):
            # If user defines a custom inference timestep, directly assign it.
            assert len(inference_timesteps) == num_inference_timesteps
            self.timesteps = torch.tensor(inference_timesteps, device=self.device, dtype=torch.int)
        elif inference_timesteps == "trailing":
            # Example 20 steps: [999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 499, 449, 399, 349, 299, 249, 199, 149,  99,  49]
            self.timesteps = torch.arange(num_train_timesteps - 1, -1, -num_train_timesteps / num_inference_timesteps, device=self.device).round().int()
        elif inference_timesteps == "linspace":
            # Example 20 steps: [999, 946, 894, 841, 789, 736, 684, 631, 578, 526, 473, 421, 368, 315, 263, 210, 158, 105,  53,   0]
            self.timesteps = torch.linspace(0, num_train_timesteps - 1, num_inference_timesteps, device=self.device).round().int().flip(0)
        elif inference_timesteps == "leading":
            # Original SD and DDIM paper may have a bug: <https://github.com/huggingface/diffusers/issues/2585>
            # The inference timestep does not start from 999.
            # Example 20 steps: [950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100,  50,   0]
            self.timesteps = torch.arange(0, num_train_timesteps, num_train_timesteps // num_inference_timesteps, device=self.device, dtype=torch.int).flip(0)
        else:
            raise NotImplementedError

    def reset(self):
        pass

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timesteps].reshape(-1, *([1] * (original_samples.ndim - 1)))
        return alpha_prod_t ** (0.5) * original_samples + (1 - alpha_prod_t) ** (0.5) * noise

    def convert_output(
        self,
        model_output: torch.Tensor,
        model_output_type: str,
        sample: torch.Tensor,
        timesteps: Union[torch.Tensor, int]
    ) -> SchedulerConversionOutput:
        assert model_output_type in self.prediction_types

        alpha_prod_t = self.alphas_cumprod[timesteps].reshape(-1, *([1] * (sample.ndim - 1)))
        beta_prod_t = 1 - alpha_prod_t

        if model_output_type == "epsilon":
            pred_epsilon = model_output
            pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)
            pred_velocity = alpha_prod_t ** (0.5) * pred_epsilon - (1 - alpha_prod_t) ** (0.5) * pred_original_sample
        elif model_output_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            pred_velocity = alpha_prod_t ** (0.5) * pred_epsilon - (1 - alpha_prod_t) ** (0.5) * pred_original_sample
        elif model_output_type == "v_prediction":
            pred_velocity = model_output
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError("Unknown prediction type")

        return SchedulerConversionOutput(
            pred_epsilon=pred_epsilon,
            pred_original_sample=pred_original_sample,
            pred_velocity=pred_velocity)

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.FloatTensor:
        alpha_prod_t = self.alphas_cumprod[timesteps].reshape(-1, *([1] * (sample.ndim - 1)))
        return alpha_prod_t ** (0.5) * noise - (1 - alpha_prod_t) ** (0.5) * sample
