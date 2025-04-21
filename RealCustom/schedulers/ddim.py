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

from .base import *


class DDIMScheduler(Scheduler):
    def step(
        self,
        model_output: torch.Tensor,
        model_output_type: str,
        timestep: Union[torch.Tensor, int],
        sample: torch.Tensor,
        eta: float = 0.0,
        clip_sample: bool = False,
        dynamic_threshold: Optional[float] = None,
        variance_noise: Optional[torch.Tensor] = None,
    ) -> SchedulerStepOutput:
        # 1. get previous step value (t-1)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.int)

        idx = timestep.reshape(-1, 1).eq(self.timesteps.reshape(1, -1)).nonzero()[:, 1]
        prev_timestep = self.timesteps[idx.add(1).clamp_max(self.num_inference_timesteps - 1)]

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].reshape(-1, *([1] * (sample.ndim - 1)))
        alpha_prod_t_prev = torch.where(idx < self.num_inference_timesteps - 1, self.alphas_cumprod[prev_timestep], self.final_alpha_cumprod).reshape(-1, *([1] * (sample.ndim - 1)))
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. compute predicted original sample from predicted noise also called
        model_output_conversion = self.convert_output(model_output, model_output_type, sample, timestep)
        pred_original_sample = model_output_conversion.pred_original_sample
        pred_epsilon = model_output_conversion.pred_epsilon

        # 4. Clip or threshold "predicted x_0"
        if clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
            pred_epsilon = self.convert_output(pred_original_sample, "sample", sample, timestep).pred_epsilon

        if dynamic_threshold is not None:
            # Dynamic thresholding in https://arxiv.org/abs/2205.11487
            dynamic_max_val = pred_original_sample \
                .flatten(1) \
                .abs() \
                .float() \
                .quantile(dynamic_threshold, dim=1) \
                .type_as(pred_original_sample) \
                .clamp_min(1) \
                .view(-1, *([1] * (pred_original_sample.ndim - 1)))
            pred_original_sample = pred_original_sample.clamp(-dynamic_max_val, dynamic_max_val) / dynamic_max_val
            pred_epsilon = self.convert_output(pred_original_sample, "sample", sample, timestep).pred_epsilon

        # 5. compute variance: "sigma_t(η)" -> see formula (16) from https://arxiv.org/pdf/2010.02502.pdf
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # 8. add "random noise" if needed.
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output)
            prev_sample = prev_sample + std_dev_t * variance_noise

        return SchedulerStepOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample)
