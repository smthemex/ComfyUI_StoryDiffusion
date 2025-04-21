import torch
from PIL import Image
from .pipeline import InstantCharacterFluxPipeline

# Step 1 Load base model and adapter
ip_adapter_path = 'checkpoints/instantcharacter_ip-adapter.bin'
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
seed = 123456
pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
)

# Step 2 Load reference image
ref_image_path = 'assets/girl.jpg'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

# Step 3 Inference without style
prompt = "A girl is playing a guitar in street"
image = pipe(
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux_instantcharacter.png")


# Step 3 Inference with style
lora_file_path = 'checkpoints/style_lora/ghibli_style.safetensors'
trigger = 'ghibli style'
prompt = "A girl is playing a guitar in street"
image = pipe.with_style_lora(
    lora_file_path=lora_file_path,
    trigger=trigger,
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux_instantcharacter_style_ghibli.png")


# Step 3 Inference with style
lora_file_path = 'checkpoints/style_lora/Makoto_Shinkai_style.safetensors'
trigger = 'Makoto Shinkai style'
prompt = "A girl is playing a guitar in street"
image = pipe.with_style_lora(
    lora_file_path=lora_file_path,
    trigger=trigger,
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux_instantcharacter_style_Makoto.png")


















