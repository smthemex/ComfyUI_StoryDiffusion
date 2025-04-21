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

import gradio as gr

from .pipeline import RealCustomInferencePipeline

def create_demo():
    pipeline = RealCustomInferencePipeline(
        unet_config="configs/realcustom_sigdino_highres.json",
        unet_checkpoint="ckpts/sdxl/unet/sdxl-unet.bin",
        realcustom_checkpoint="ckpts/realcustom/RealCustom_highres.pth",
        vae_config="ckpts/sdxl/vae/sdxl.json",
        vae_checkpoint="ckpts/sdxl/vae/sdxl-vae.pth",
    )

    badges_text = r"""
    <div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
    <a href="https://corleone-huang.github.io/RealCustom_plus_plus/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-RealCustom-yellow"></a> 
    <a href="https://arxiv.org/pdf/2408.09744?"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-RealCustom-b31b1b.svg"></a>
    </div>
    """.strip()

    with gr.Blocks() as demo:
        gr.Markdown(f"# RealCustom")
        gr.Markdown(badges_text)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="")
                target_phrase = gr.Textbox(label="Target Phrase", value="")
                with gr.Row():
                    image_prompt = gr.Image(label="Ref Img", visible=True, interactive=True, type="pil")

                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(512, 2048, 1024, step=16, label="Gneration Width")
                        height = gr.Slider(512, 2048, 1024, step=16, label="Gneration Height")

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        guidance = gr.Slider(1.0, 15, 3.5, step=0.5, label="Guidance Scale", interactive=True)
                        mask_scope = gr.Slider(0.05, 1.0, 0.2, step=0.05, label="Mask Scope", interactive=True)
                        seed = gr.Number(0, label="Seed (-1 for random)")
                        num = gr.Number(4, label="Generation Number")
                        new_unet_local_path = gr.Textbox(label="New Unet Local Path", value="")
                        new_realcustom_local_path = gr.Textbox(label="New RealCustom Local Path", value="")

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                output_mask = gr.Image(label="Guidance Mask")

            inputs = [
                prompt, image_prompt, target_phrase, 
                height, width, guidance, seed, num, 
                mask_scope, 
                new_unet_local_path, new_realcustom_local_path,
            ]
            generate_btn.click(
                fn=pipeline.generation,
                inputs=inputs,
                outputs=[output_image, output_mask],
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name='0.0.0.0', server_port=7860)