import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, prompt: str,if_repo:bool,device,h,w) -> dict[str, Tensor]:

    h=2 * math.ceil(h / 16)
    w=2 * math.ceil(w / 16)
    #bs, c, h, w = img.shape
    # if bs == 1 and not isinstance(prompt, str):
    #     bs = len(prompt)
    bs=1
    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # if img.shape[0] == 1 and bs > 1:
    #     img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    
    if if_repo:
        if isinstance(prompt, str):
            prompt = [prompt]
    txt = t5(prompt)
 
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    
    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
    
    #print(txt.shape,vec.shape) #torch.Size([1, 256, 4096]) torch.Size([1, 768])
   
    if if_repo:
        return {

            "img_ids": img_ids.to(device),
            "txt": txt.to(device),
            "txt_ids": txt_ids.to(device, ),
            "vec": vec.to(device),
        }
    else:
        return {
            
            "img_ids": img_ids.to(device, torch.bfloat16),
            "txt": txt.to(device, torch.bfloat16),
            "txt_ids": txt_ids.to(device, torch.bfloat16),
            "vec": vec.to(device, torch.bfloat16),
        }
    


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    id_weight=1.0,
    id=None,
    start_step=0,
    uncond_id=None,
    true_cfg=1.0,
    timestep_to_start_cfg=1,
    neg_txt=None,
    neg_txt_ids=None,
    neg_vec=None,
    aggressive_offload=False,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    use_true_cfg = abs(true_cfg - 1.0) > 1e-2
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            id=id if i >= start_step else None,
            id_weight=id_weight,
            aggressive_offload=aggressive_offload,
        )

        if use_true_cfg and i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                id=uncond_id if i >= start_step else None,
                id_weight=id_weight,
                aggressive_offload=aggressive_offload,
            )
            pred = neg_pred + true_cfg * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
