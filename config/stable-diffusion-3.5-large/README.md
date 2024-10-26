---
license: other
license_name: stabilityai-nc-research-community
license_link: LICENSE
tags:
- text-to-image
- stable-diffusion
extra_gated_prompt: >-
  By clicking "Agree", you agree to the [License
  Agreement](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE)
  and acknowledge Stability AI's [Privacy
  Policy](https://stability.ai/privacy-policy).
extra_gated_fields:
  Name: text
  Email: text
  Country: country
  Organization or Affiliation: text
  Receive email updates and promotions on Stability AI products, services, and research?:
    type: select
    options:
    - 'Yes'
    - 'No'
  I acknowledge that this model is for non-commercial use only unless I acquire a separate license from Stability AI: checkbox
language:
- en
pipeline_tag: text-to-image
---
# Stable Diffusion 3 Medium
![sd3 demo images](sd3demo.jpg)

## Model

![mmdit](mmdit.png)

[Stable Diffusion 3 Medium](stability.ai/news/stable-diffusion-3-medium) is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features greatly improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

For more technical details, please refer to the [Research paper](https://stability.ai/news/stable-diffusion-3-research-paper).

Please note: this model is released under the Stability Non-Commercial Research Community License. For a Creator License or an Enterprise License visit Stability.ai or [contact us](https://stability.ai/license) for commercial licensing details.

### Model Description

- **Developed by:** Stability AI
- **Model type:** MMDiT text-to-image generative model
- **Model Description:** This is a model that can be used to generate images based on text prompts. It is a Multimodal Diffusion Transformer
(https://arxiv.org/abs/2403.03206) that uses three fixed, pretrained text encoders 
([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip), [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main) and [T5-xxl](https://huggingface.co/google/t5-v1_1-xxl))

### License

- **Non-commercial Use:** Stable Diffusion 3 Medium is released under the [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE). The model is free to use for non-commercial purposes such as academic research.
- **Commercial Use**: This model is not available for commercial use without a separate commercial license from Stability. We encourage professional artists, designers, and creators to use our Creator License. Please visit https://stability.ai/license to learn more.


### Model Sources

For local or self-hosted use, we recommend [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for inference.

Stable Diffusion 3 Medium is available on our [Stability API Platform](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post). 

Stable Diffusion 3 models and workflows are available on [Stable Assistant](https://stability.ai/stable-assistant) and on Discord via [Stable Artisan](https://stability.ai/stable-artisan). 

- **ComfyUI:** https://github.com/comfyanonymous/ComfyUI
- **StableSwarmUI:** https://github.com/Stability-AI/StableSwarmUI
- **Tech report:** https://stability.ai/news/stable-diffusion-3-research-paper
- **Demo:** https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium


## Training Dataset

We used synthetic data and filtered publicly available data to train our models. The model was pre-trained on 1 billion images. The fine-tuning data includes 30M high-quality aesthetic images focused on specific visual content and style, as well as 3M preference data images.

## Using with Diffusers

Make sure you upgrade to the latest version of `diffusers`: `pip install -U diffusers`. And then you can run:

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
	"A cat holding a sign that says hello world",
	negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image
```

Refer to [the documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_3) for more details on optimization and image-to-image support. 

## Uses

### Intended Uses

Intended uses include the following: 
* Generation of artworks and use in design and other artistic processes.
* Applications in educational or creative tools.
* Research on generative models, including understanding the limitations of generative models.

All uses of the model should be in accordance with our [Acceptable Use Policy](https://stability.ai/use-policy).

### Out-of-Scope Uses

The model was not trained to be factual or true representations of people or events.  As such, using the model to generate such content is out-of-scope of the abilities of this model.

## Safety

As part of our safety-by-design and responsible AI deployment approach, we implement safety measures throughout the development of our models, from the time we begin pre-training a model to the ongoing development, fine-tuning, and deployment of each model. We have implemented a number of safety mitigations that are intended to reduce the risk of severe harms, however we recommend that developers conduct their own testing and apply additional mitigations based on their specific use cases.  
For more about our approach to Safety, please visit our [Safety page](https://stability.ai/safety).

### Evaluation Approach

Our evaluation methods include structured evaluations and internal and external red-teaming testing for specific, severe harms such as child sexual abuse and exploitation, extreme violence, and gore, sexually explicit content, and non-consensual nudity.  Testing was conducted primarily in English and may not cover all possible harms.  As with any model, the model may, at times, produce inaccurate, biased or objectionable responses to user prompts. 

### Risks identified and mitigations:

* Harmful content:  We have used filtered data sets when training our models and implemented safeguards that attempt to strike the right balance between usefulness and preventing harm. However, this does not guarantee that all possible harmful content has been removed. The model may, at times, generate toxic or biased content.  All developers and deployers should exercise caution and implement content safety guardrails based on their specific product policies and application use cases.
* Misuse: Technical limitations and developer and end-user education can help mitigate against malicious applications of models. All users are required to adhere to our Acceptable Use Policy, including when applying fine-tuning and prompt engineering mechanisms. Please reference the Stability AI Acceptable Use Policy for information on violative uses of our products.
* Privacy violations: Developers and deployers are encouraged to adhere to privacy regulations with techniques that respect data privacy.

### Contact

Please report any issues with the model or contact us:

* Safety issues:  safety@stability.ai
* Security issues:  security@stability.ai
* Privacy issues:  privacy@stability.ai 
* License and general: https://stability.ai/license
* Enterprise license: https://stability.ai/enterprise
