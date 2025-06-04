# import spaces
import os
import time
import torch
# import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora

BASE_PATH = "black-forest-labs/FLUX.1-dev"
LOCAL_LORA_DIR = "./LoRAs"          
CUSTOM_LORA_DIR = "./Custom_LoRAs"  
os.makedirs(LOCAL_LORA_DIR, exist_ok=True)
os.makedirs(CUSTOM_LORA_DIR, exist_ok=True)

print("downloading OmniConsistency base LoRA …")
omni_consistency_path = hf_hub_download(
    repo_id="showlab/OmniConsistency",
    filename="OmniConsistency.safetensors",
    local_dir="./Model"
)

print("loading base pipeline …")
pipe = FluxPipeline.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16
).to("cuda")
set_single_lora(pipe.transformer, omni_consistency_path,
                lora_weights=[1], cond_size=512)

def download_all_loras():
    lora_names = [
        "3D_Chibi", "American_Cartoon", "Chinese_Ink", "Clay_Toy",
        "Fabric", "Ghibli", "Irasutoya", "Jojo", "LEGO", "Line",
        "Macaron", "Oil_Painting", "Origami", "Paper_Cutting",
        "Picasso", "Pixel", "Poly", "Pop_Art", "Rick_Morty",
        "Snoopy", "Van_Gogh", "Vector"
    ]
    for name in lora_names:
        hf_hub_download(
            repo_id="showlab/OmniConsistency",
            filename=f"LoRAs/{name}_rank128_bf16.safetensors",
            local_dir=LOCAL_LORA_DIR,
        )
download_all_loras()

def clear_cache(transformer):
    for _, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# @spaces.GPU()
def generate_image(
    lora_name,        
    custom_repo_id,   
    prompt,
    uploaded_image,
    width, height,
    guidance_scale,
    num_inference_steps,
    seed
):
    width, height = int(width), int(height)
    generator = torch.Generator("cpu").manual_seed(seed)

    if custom_repo_id and custom_repo_id.strip():
        repo_id = custom_repo_id.strip()
        try:
            files = list_repo_files(repo_id)
            print("using custom LoRA from:", repo_id)
            safetensors_files = [f for f in files if f.endswith(".safetensors")]
            print("found safetensors files:", safetensors_files)
            if not safetensors_files:
                raise ValueError("No .safetensors files were found in this repo")
            fname = safetensors_files[0]
            lora_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=CUSTOM_LORA_DIR,
            )
        except Exception as e:
            raise gr.Error(f"Load custom LoRA failed: {e}")
    else:
        lora_path = os.path.join(
            f"{LOCAL_LORA_DIR}/LoRAs", f"{lora_name}_rank128_bf16.safetensors"
        )

    pipe.unload_lora_weights()
    try:
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path)
        )
    except Exception as e:
        raise gr.Error(f"Load LoRA failed: {e}")

    spatial_image  = [uploaded_image.convert("RGB")]
    subject_images = []
    start = time.time()
    out_img = pipe(
        prompt,
        height=(height // 8) * 8,
        width=(width  // 8) * 8,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator,
        spatial_images=spatial_image,
        subject_images=subject_images,
        cond_size=512,
    ).images[0]
    print(f"inference time: {time.time()-start:.2f}s")

    clear_cache(pipe.transformer)
    return uploaded_image, out_img

# =============== Gradio UI ===============
def create_interface():
    demo_lora_names = [
        "3D_Chibi", "American_Cartoon", "Chinese_Ink", "Clay_Toy",
        "Fabric", "Ghibli", "Irasutoya", "Jojo", "LEGO", "Line",
        "Macaron", "Oil_Painting", "Origami", "Paper_Cutting",
        "Picasso", "Pixel", "Poly", "Pop_Art", "Rick_Morty",
        "Snoopy", "Van_Gogh", "Vector"
    ]

    def update_trigger_word(lora_name, prompt):
      for name in demo_lora_names:
        trigger = " ".join(name.split("_")) + " style,"
        prompt = prompt.replace(trigger, "")
      new_trigger = " ".join(lora_name.split("_"))+ " style,"
      return new_trigger + prompt

    # Example data
    examples = [
        ["3D_Chibi", "", "3D Chibi style, Two smiling colleagues enthusiastically high-five in front of a whiteboard filled with technical notes about multimodal learning, reflecting a moment of success and collaboration at OpenAI.", 
        Image.open("./test_imgs/00.png"), 680, 1024, 3.5, 24, 42],
        ["Clay_Toy", "", "Clay Toy style, Three team members from OpenAI are gathered around a laptop in a cozy, festive setting, with holiday decorations in the background; one waves cheerfully while the others engage in light conversation, reflecting a relaxed and collaborative atmosphere.", 
        Image.open("./test_imgs/01.png"), 560, 1024, 3.5, 24, 42],
        ["American_Cartoon", "", "American Cartoon style, In a dramatic and comedic moment from a classic Chinese film, an intense elder with a white beard and red hat grips a younger man, declaring something with fervor, while the subtitle at the bottom reads 'I want them all' — capturing both tension and humor.",  
        Image.open("./test_imgs/02.png"), 568, 1024, 3.5, 24, 42],
        ["Origami", "", "Origami style, A thrilled fan wearing a Portugal football kit poses energetically with a smiling Cristiano Ronaldo, who gives a thumbs-up, as they stand side by side in a casual, cheerful moment—capturing the excitement of meeting a football legend.", 
        Image.open("./test_imgs/03.png"), 768, 672, 3.5, 24, 42],
        ["Vector", "", "Vector style, A man glances admiringly at a passing woman, while his girlfriend looks at him in disbelief, perfectly capturing the theme of shifting attention and misplaced priorities in a humorous, relatable way.", 
        Image.open("./test_imgs/04.png"), 512, 1024, 3.5, 24, 42]
    ]

    header = """
<div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
<a href="https://arxiv.org/abs/2505.18445"><img src="https://img.shields.io/badge/ariXv-2505.18445-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/showlab/OmniConsistency"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://github.com/showlab/OmniConsistency"><img src="https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github&" alt="GitHub"></a>
</div>
"""

    with gr.Blocks() as demo:
        gr.Markdown("# OmniConsistency LoRA Image Generation")
        gr.Markdown("Select a LoRA, enter a prompt, and upload an image to generate a new image with OmniConsistency.")
        gr.HTML(header)

        with gr.Row():
            with gr.Column(scale=1):
               image_input = gr.Image(type="pil", label="Upload Image")
               prompt_box = gr.Textbox(label="Prompt",
                                        value="3D Chibi style,",
                                        info="Remember to include the necessary trigger words if you're using a custom LoRA."
                )
               lora_dropdown = gr.Dropdown(
                    demo_lora_names, label="Select built-in LoRA")
               custom_repo_box = gr.Textbox(
                    label="Enter Custom LoRA",
                    placeholder="LoRA Hugging Face path (e.g., 'username/repo_name')",
                    info="If you want to use a custom LoRA, enter its Hugging Face repo ID here and built-in LoRA will be Overridden. Leave empty to use built-in LoRAs. [Check the list of FLUX LoRAs](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev)"
                )
               gen_btn = gr.Button("Generate")
            with gr.Column(scale=1):
                output_image = gr.ImageSlider(label="Generated Image")
        with gr.Accordion("Advanced Options", open=False):
          height_box = gr.Textbox(value="1024", label="Height")
          width_box  = gr.Textbox(value="1024", label="Width")
          guidance_slider = gr.Slider(
              0.1, 20, value=3.5, step=0.1, label="Guidance Scale")
          steps_slider = gr.Slider(
              1, 50, value=25, step=1, label="Inference Steps")
          seed_slider = gr.Slider(
              1, 2_147_483_647, value=42, step=1, label="Seed")

        lora_dropdown.select(fn=update_trigger_word, inputs=[lora_dropdown,prompt_box], 
                             outputs=prompt_box)

        gr.Examples(
            examples=examples,
            inputs=[lora_dropdown, custom_repo_box, prompt_box, image_input,
                    height_box, width_box, guidance_slider, steps_slider, seed_slider],
            outputs=output_image,
            fn=generate_image,
            cache_examples=False,
            label="Examples"
        )

        gen_btn.click(
            fn=generate_image,
            inputs=[lora_dropdown, custom_repo_box, prompt_box, image_input,
                    width_box, height_box, guidance_slider, steps_slider, seed_slider],
            outputs=output_image
        )
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(ssr_mode=False)