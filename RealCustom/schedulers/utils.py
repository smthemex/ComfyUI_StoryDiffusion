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

import math
from typing import List

import torch


def get_betas(name: str, num_steps: int = 1000, shift_snr: float = 1, terminal_pure_noise: bool = False):
    # Get betas
    max_beta = 1 if terminal_pure_noise else 0.999
    if name == "squared_linear":
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, num_steps) ** 2
    elif name == "cosine":
        betas = get_cosine_betas(num_steps, max_beta=max_beta)
    elif name == "alphas_cumprod_linear":
        betas = get_alphas_cumprod_linear_betas(num_steps, max_beta=max_beta)
    elif name == "sigmoid":
        betas = get_sigmoid_betas(num_steps, max_beta=max_beta, square=True, slop=0.7)
    else:
        raise NotImplementedError

    # Shift snr
    betas = shift_betas_by_snr_factor(betas, shift_snr)

    # Ensure terminal pure noise
    # Only non-cosine schedule uses rescale, cosine schedule can directly set max_beta=1 to ensure temrinal pure noise.
    if name == "squared_linear" and terminal_pure_noise:
        betas = rescale_betas_to_ensure_terminal_pure_noise(betas)

    return betas


def validate_betas(betas: List[float]) -> bool:
    """
    Validate betas is monotonic and within 0 to 1 range, i.e. 0 < beta_{t-1} < beta_{t} <= 1

    Args:
        betas (List[float]): betas

    Returns:
        bool: True if betas is correct
    """
    return all(b1 < b2 for b1, b2 in zip(betas, betas[1:])) and betas[0] > 0 and betas[-1] <= 1


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar_fn, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    if not validate_betas(betas):
        import logging
        logging.warning("No feasible betas for given alpha bar")
    return torch.tensor(betas, dtype=torch.float32)


def get_cosine_betas(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    def alpha_bar_fn(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
    return betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar_fn, max_beta)


def get_sigmoid_betas(num_diffusion_timesteps, max_beta, square=False, slop=1):
    def alpha_bar_fn(t):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x * slop))
        s = 6  # (-6, 6) from geodiff
        vb = sigmoid(-s)
        ve = sigmoid(s)
        return ((sigmoid(s - t * 2 * s) - vb) / (ve - vb))**(1 if not square else 2)
    return betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar_fn, max_beta)


def get_alphas_cumprod_linear_betas(num_diffusion_timesteps, max_beta):
    def alpha_bar_fn(t):
        return 1 - t
    return betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar_fn, max_beta=max_beta)


def shift_betas_by_snr_factor(betas: torch.Tensor, factor: float) -> torch.Tensor:
    if factor == 1.0:
        return betas
    # Convert betas to snr
    alphas = 1 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    snr = alphas_cumprod / (1 - alphas_cumprod)
    # Shift snr
    snr *= factor
    # Convert snr to betas
    alphas_cumprod = snr / (1 + snr)
    alphas = torch.cat(
        [alphas_cumprod[0:1], alphas_cumprod[1:] / alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas


def rescale_betas_to_ensure_terminal_pure_noise(betas: torch.Tensor) -> torch.Tensor:
    # Convert betas to alphas_cumprod_sqrt
    alphas = 1 - betas
    alphas_cumprod = alphas.cumprod(0)
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()
    # Rescale alphas_cumprod_sqrt such that alphas_cumprod_sqrt[0] remains unchanged but alphas_cumprod_sqrt[-1] = 0
    alphas_cumprod_sqrt = (alphas_cumprod_sqrt - alphas_cumprod_sqrt[-1]) / (
        alphas_cumprod_sqrt[0] - alphas_cumprod_sqrt[-1]) * alphas_cumprod_sqrt[0]
    # Convert alphas_cumprod_sqrt to betas
    alphas_cumprod = alphas_cumprod_sqrt ** 2
    alphas = torch.cat(
        [alphas_cumprod[0:1], alphas_cumprod[1:] / alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas
