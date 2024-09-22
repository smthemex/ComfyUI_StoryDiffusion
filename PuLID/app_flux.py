import time
import torch
from einops import rearrange
from PIL import Image

from .flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from .flux.util import load_ae, load_clip, load_flow_model, load_t5, load_flow_model_quintized,SamplingOptions
from .pulid.pipeline_flux import PuLIDPipeline


def get_models(name: str,ckpt_path, device: torch.device, offload: bool,quantized_mode):
    t5 = load_t5(name,quantized_mode,device, max_length=128)
    clip = load_clip(name,quantized_mode,device)
    if quantized_mode=="fp8" or quantized_mode=="nf4":
        model = load_flow_model_quintized(name, ckpt_path,quantized_mode,device="cpu" if offload else device)
    else:
        model = load_flow_model(name, ckpt_path,device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, ckpt_path,device: str, offload: bool,aggressive_offload: bool, pretrained_model,quantized_mode,clip_vision_path):
        self.device = torch.device(device)
        self.offload = offload
        self.model_name = model_name
        self.aggressive_offload= aggressive_offload
        self.ckpt_path=ckpt_path
        self.clip_vision_path=clip_vision_path
        self.quantized_mode=quantized_mode
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,self.ckpt_path,
            device=self.device,
            offload=self.offload,quantized_mode=self.quantized_mode,
        )
        self.pulid_model = PuLIDPipeline(self.model, device, self.clip_vision_path,weight_dtype=torch.bfloat16)
        self.pulid_model.load_pretrain(pretrained_model)

    @torch.inference_mode()
    def generate_image(
            self,
            width,
            height,
            num_steps,
            start_step,
            guidance,
            seed,
            prompt,
            id_embeddings = None,
            uncond_id_embeddings=None,
            id_weight=1.0,
            neg_prompt="",
            true_cfg=1.0,
            timestep_to_start_cfg=1,
            max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None
        
        if isinstance(prompt,str):
            prompt_list=[prompt]
        elif isinstance(prompt,list):
            prompt_list=prompt
        else:
            raise "prompt must be list or str"

        img_output = []
        for i,prompt in enumerate(prompt_list):
            opts = SamplingOptions(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance=guidance,
                seed=seed,
            )
            
            if opts.seed is None:
                opts.seed = torch.Generator(device="cpu").seed()
            print(f"Generating '{opts.prompt}' with seed {opts.seed}")
            t0 = time.perf_counter()
            
            use_true_cfg = abs(true_cfg - 1.0) > 1e-2

            # prepare input
            x = get_noise(
                1,
                opts.height,
                opts.width,
                device=self.device,
                dtype=torch.bfloat16,
                seed=opts.seed,
            )
            timesteps = get_schedule(
                opts.num_steps,
                x.shape[-1] * x.shape[-2] // 4,
                shift=True,
            )
            
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
            inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None
            
            # offload TEs to CPU, load model to gpu
            
            if self.offload:
                self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
                torch.cuda.empty_cache()
                if self.aggressive_offload:
                    self.model.components_to_gpu()
                else:
                    self.model = self.model.to(self.device)
            
            # denoise initial noise
            print("start denoise...")
            x = denoise(
                self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
                start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=inp_neg["txt"] if use_true_cfg else None,
                neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
                neg_vec=inp_neg["vec"] if use_true_cfg else None,
                aggressive_offload=self.aggressive_offload,
            )
            
            # offload model, load autoencoder to gpu
            print("start decoder...")
            if self.offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                self.ae.decoder.to(x.device)
            
            # decode latents to pixel space
            x = unpack(x.float(), opts.height, opts.width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)
            
            if self.offload:
                self.ae.decoder.cpu()
                torch.cuda.empty_cache()
            
            t1 = time.perf_counter()
            
            print(f"Done in {t1 - t0:.1f}s.")
            
            # bring into PIL format
            x = x.clamp(-1, 1)
            # x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")
            
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            # return img, str(opts.seed), self.pulid_model.debug_img_list
            img_output.append(img)
        return img_output
