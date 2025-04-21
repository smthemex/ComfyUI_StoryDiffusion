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


class DPMSolverSingleStepScheduler(Scheduler):
    def __init__(
        self,
        # Generic scheduler settings
        num_train_timesteps: int,
        num_inference_timesteps: int,
        betas: torch.Tensor,
        inference_timesteps: Union[str, List[int]] = "trailing",
        set_alpha_to_one: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        # DPM scheduler settings
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        solver_order: int = 2,
        lower_order_final: bool = True,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            num_inference_timesteps=num_inference_timesteps,
            betas=betas,
            inference_timesteps=inference_timesteps,
            set_alpha_to_one=set_alpha_to_one,
            device=device,
            dtype=dtype,
        )

        self.solver_order = solver_order
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.algorithm_type = algorithm_type

        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        self.reset()

    def reset(self):
        self.model_outputs = [None] * self.solver_order
        self.sample = None
        self.order_list = self.get_order_list()
        self.last_step_index = None

    def get_order_list(self) -> List[int]:
        steps = self.num_inference_timesteps
        order = self.solver_order
        # First step must be order 1
        # Second step must be order 1 in case of terminal zero SNR
        orders = [1] + [(i % order) + 1 for i in range(steps - 1)] + [1]
        # Last step should be order 1 for better quality.
        if self.lower_order_final:
            orders[-1] = 1
        return orders

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        return x_t

    def singlestep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s1 = self.alpha_t[t], self.alpha_t[s1]
        sigma_t, sigma_s1 = self.sigma_t[t], self.sigma_t[s1]
        h, h_0 = lambda_t - lambda_s1, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m1, (1.0 / r0) * (m0 - m1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s1) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s1) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s1) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s1) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        return x_t

    def singlestep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s2 = self.alpha_t[t], self.alpha_t[s2]
        sigma_t, sigma_s2 = self.sigma_t[t], self.sigma_t[s2]
        h, h_0, h_1 = lambda_t - lambda_s2, lambda_s0 - lambda_s2, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m2
        D1_0, D1_1 = (1.0 / r1) * (m1 - m2), (1.0 / r0) * (m0 - m2)
        D1 = (r0 * D1_0 - r1 * D1_1) / (r0 - r1)
        D2 = 2.0 * (D1_1 - D1_0) / (r0 - r1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s2) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1_1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s2) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                    - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s2) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1_1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s2) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
                )
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        model_output_type: str,
        timestep: int,
        sample: torch.FloatTensor,
    ) -> SchedulerStepOutput:

        step_index = (self.timesteps == timestep).nonzero().item()

        # Check if this step is the follow-up of the previous step.
        # If not, then we reset and treat it as a new run.
        if self.last_step_index and self.last_step_index != step_index - 1:
            self.reset()
        self.last_step_index = step_index

        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        model_output_convert = self.convert_output(model_output, model_output_type, sample, timestep)

        if self.algorithm_type == "dpmsolver++":
            model_output = model_output_convert.pred_original_sample
        else:
            model_output = model_output_convert.pred_epsilon

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        order = self.order_list[step_index]

        #  For img2img denoising might start with order>1 which is not possible
        #  In this case make sure that the first two steps are both order=1
        while self.model_outputs[-order] is None:
            order -= 1

        # For single-step solvers, we use the initial value at each time with order = 1.
        if order == 1:
            self.sample = sample

        timestep_list = [self.timesteps[step_index - i] for i in range(order - 1, 0, -1)] + [timestep]

        if order == 1:
            prev_sample = self.dpm_solver_first_order_update(self.model_outputs[-1], timestep_list[-1], prev_timestep, self.sample)
        elif order == 2:
            prev_sample = self.singlestep_dpm_solver_second_order_update(self.model_outputs, timestep_list, prev_timestep, self.sample)
        elif order == 3:
            prev_sample = self.singlestep_dpm_solver_third_order_update(self.model_outputs, timestep_list, prev_timestep, self.sample)
        else:
            raise NotImplementedError

        return SchedulerStepOutput(
            prev_sample=prev_sample,
            pred_original_sample=model_output_convert.pred_original_sample
        )
