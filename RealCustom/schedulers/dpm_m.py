# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

from typing import List, Optional, Union

import numpy as np
import torch

from .base import *


class DPMSolverMultistepScheduler(Scheduler):
    def __init__(
        self,
        # Generic scheduler settings
        num_inference_timesteps: int,
        betas: torch.Tensor,
        num_train_timesteps: int = 1000,
        inference_timesteps: Union[str, List[str]] = "trailing",
        set_alpha_to_one: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        # DPM scheduler settings
        solver_order: int = 2,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        use_karras_sigmas: bool = False,
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

        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        if use_karras_sigmas:
            log_sigmas = torch.log(sigmas)
            sigmas = self._convert_to_karras(
                in_sigmas=sigmas, num_inference_timesteps=num_inference_timesteps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas)
                                 for sigma in sigmas]).round()
            timesteps = np.flip(timesteps).copy().astype(np.int64)
            self.timesteps = torch.from_numpy(timesteps).to(device)
            sigmas = torch.from_numpy(sigmas).to(device)
        self.sigmas = sigmas

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++", "deis"]:
            raise NotImplementedError(
                f"{algorithm_type} does is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun", "logrho", "bh1", "bh2"]:
            raise NotImplementedError(
                f"{solver_type} does is not implemented for {self.__class__}")

        # setable values
        self.reset()

    def reset(self):
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(
            axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.FloatTensor, num_inference_timesteps) -> torch.FloatTensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_timesteps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - \
                (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - \
                (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (alpha_t / alpha_s0) * sample
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
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_timesteps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(
            self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) -
             1) and self.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) -
             2) and self.lower_order_final and len(self.timesteps) < 15
        )

        model_output_convert = self.convert_output(
            model_output, model_output_type=model_output_type, sample=sample, timesteps=timestep)
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            model_output = model_output_convert.pred_original_sample
        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            model_output = model_output_convert.pred_epsilon

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = torch.randn_like(
                model_output, device=model_output.device, dtype=model_output.dtype)
        else:
            noise = None

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise=noise
            )
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
            )
        else:
            timestep_list = [self.timesteps[step_index - 2],
                             self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        return SchedulerStepOutput(prev_sample=prev_sample, pred_original_sample=model_output_convert.pred_original_sample)
