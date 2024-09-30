
import torch
import numpy as np
from PIL import Image
import node_helpers
from nodes import common_ksampler

def phi2tensor(img):
    tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor

def tensor_to_image(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image
class CFGenerator:
    def __init__(self, model,clip,ae,model_type, device: str):
        self.device = torch.device(device)
        self.model=model
        self.ae=ae
        self.clip=clip
        self.model_type=model_type

    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        cfg,
        guidance,
        seed,
        prompt,
        negative_prompt,
        cf_scheduler,
        denoise,
        image,
    ):
        if isinstance(prompt,str):
            text=prompt
        elif isinstance(prompt,list):
            text=prompt[0] if len(prompt)==1 else str(prompt)
        else:
            raise "prompt must be str or list"
        if isinstance(negative_prompt, str):
            negative_text = negative_prompt
        elif isinstance(negative_prompt, list):
            negative_text =negative_prompt[0] if len(negative_prompt) == 1 else str(negative_prompt)
        else:
            raise "prompt must be str or list"
        
        batch_size=1
        
       
        #cf clip postive
        tokens = self.clip.tokenize(text)
        output = self.clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning=[[cond, output]]
        
        #flux GUIDANCE
        if self.model_type == "FLUX":
            postive_c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        else:
            postive_c=conditioning
            
        #cf neg
        tokens_ = self.clip.tokenize(negative_text)
        output_ = self.clip.encode_from_tokens(tokens_, return_pooled=True, return_dict=True)
        cond_ = output_.pop("cond")
        negative_c = [[cond_, output_]]
        
        if image:
            if isinstance(image,list):
                image=image[0] if len(image)==1 else None
            if isinstance(image,torch.Tensor):
                pixels=image
            elif isinstance(image,Image.Image):
                pixels=phi2tensor(image)
            elif isinstance(image, np.ndarray):
                init_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                pixels = init_image.unsqueeze(0)
            else:
                raise "image input error"
            #print(type(image),image,pixels.shape,type(pixels))
            t = self.ae.encode(pixels[:, :, :, :3])
        else:
            if self.model_type == "FLUX":
                t = torch.ones([batch_size, 16, height // 8, width // 8], device=self.device) * 0.0609
            else:
                t = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
            
        latent = {"samples": t}
        
        sampler_name= cf_scheduler["name"]
        scheduler=cf_scheduler["scheduler"]
        samples=common_ksampler(self.model, seed, num_steps, cfg, sampler_name, scheduler, postive_c, negative_c, latent, denoise=denoise)
        samples_=samples[0]["samples"]
        img_out=self.ae.decode(samples_)
        #print(img_out.shape,type(img_out))
        img_pil=tensor_to_image(img_out)
        return  img_pil
        


