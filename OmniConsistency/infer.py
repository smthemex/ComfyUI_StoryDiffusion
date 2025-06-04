import time
import torch
from PIL import Image
from .src_inference.pipeline import FluxPipeline
from .src_inference.lora_helper import set_single_lora

# torch.cuda.set_device(0)

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# base_path = "black-forest-labs/FLUX.1-dev"
# pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16).to("cuda")

# set_single_lora(pipe.transformer, 
#                 "/path/to/OmniConsistency.safetensors", 
#                 lora_weights=[1], cond_size=512)

# pipe.unload_lora_weights()
# pipe.load_lora_weights("/path/to/lora_folder", 
#                        weight_name="lora_name.safetensors")

# image_path1 = "figure/test.png"
# prompt = "3D Chibi style, Three individuals standing together in the office."

# subject_images = []
# spatial_image = [Image.open(image_path1).convert("RGB")]

# width, height = 1024, 1024

# start_time = time.time()

# image = pipe(
#     prompt,
#     height=height,
#     width=width,
#     guidance_scale=3.5,
#     num_inference_steps=25,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(5),
#     spatial_images=spatial_image,
#     subject_images=subject_images,
#     cond_size=512,
# ).images[0]

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"code running time: {elapsed_time} s")

# # Clear cache after generation
# clear_cache(pipe.transformer)

# image.save("results/output.png")
