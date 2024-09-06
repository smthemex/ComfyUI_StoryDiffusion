# ComfyUI_StoryDiffusion
You can using StoryDiffusion in ComfyUI.

* [中文说明](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/README-CN.md)  
* StoryDiffusion origin From: [link](https://github.com/HVision-NKU/StoryDiffusion)  ---&---  MS-Diffusion origin From: [link](https://github.com/MS-Diffusion/MS-Diffusion)


## Updates:
**2024/09/06**   
*fix runway error,when loader single model.  

**Previous updates：**  
*The loading speed of the NF4 model is many times faster than FP8, so I recommend using the NF4 model to run Flux. I have included the workflow of NF4 in the example，
* when easy_function fill in NF4 or nf4 ,can try NF4 FLUX ,need download weights at [link](https://huggingface.co/sayakpaul/flux.1-dev-nf4/tree/main),put weight in "comfyui/models/checkpoints/";   
* Add an "easy_function" for debugging new function. This time, I have added support for "auraface" in "Photomake V2". You can enter "auraface" into the "easy_function" to test this method
* Flux fp8 single weights only support "Kijai/flux-fp8" or auto save's pt weights.
* fix some bug,Changing the loading method of Msdiffusion, reducing meaningless buttons, and removing the int4 mode of flux did not make it much faster;  
* support mps again...
* Now clip checkpoints no need diffusers_repo,you can using "clip_g.safetensors" or other base from "CLIP-ViT-bigG-14-laion2B-39B-b160k";   
* Refactoring some code about MS-diffusion,controlnet still normal quality;      
* 2 role in 1 img now using [A]...[B]... mode,  
* if first using flux repo,will automatically save the PT file on checkpoint dir(name: transform_time.pt),So you only need to run the repo and flux once separately (without completing it) to obtain the PT model, Recommend using "repo+transfomer.pt" or "repo+other fp8.safetensors" or "repo+any_name.pt(rename from transfomer )",
* 2 ways to using flux，using repo like :"black-forest-labs/FLUX.1-dev" or "X:/xxx/xxx/black-forest-labs/FLUX.1-dev"  and ckpt_name="none" in new loader node or old,or using repo like :"black-forest-labs/FLUX.1-dev" or "X:/xxx/xxx/black-forest-labs/FLUX.1-dev",and using single ckpt like "flux1-dev-fp8.safetensors";      
* Support "kolors" text2img and "kolors"ipadapter img2img,using repo like :"xxx:/xxx/xxx/Kwai-Kolors/Kolors"  (Please refer to the end of the article for detailed file combinations)  
* support photomaker V2;  
* ControlNet now uses community models.   
* The base model now has only two options: using repo input or selecting the community model...  
* To fix the bug where MS diffusion cannot run continuously in the txt2img, it is necessary to enable the "reset_txt2img" of the loading model node to be Ture;   
* Introducing Controlnet for dual character co image, supporting multi image introduction, 
* Add the function of saving and loading character models   

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt
```
if using photomakerV2,need:  
```
pip install insightface==0.7.3  or new   
```
If the module is missing, please pip install   

3 Need  model 
----
3.1.1base   
You can directly fill in the repo, such as:"stablityai/table diffusion xl base-1.0", or select the corresponding model in the local diffuser menu (provided that you have the model in the "models/diffuser" directory), or you can directly select a single SDXL community model. The priority of repo or local diffusers is higher than that of individual community models.    

Supports all SDXL based diffusion models (such as "G161222/RealVisXL_V4.0", "sd-community/sdxl-flash"）， It also supports non SD models, such as ("stablediffusionapi/sdxl-unstable-diffusers-y", playground-v2.5-1024px-aesthetic）   
When using your local SDXL monomer model (for example: Jumpernaut XL_v9-RunDiffusionPhoto_v2. safetensors), please set local_diffusers to none and download the corresponding config files to run.  

photomaker-v1.bin    [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   
photomaker-v2.bin    [link](https://huggingface.co/TencentARC/PhotoMaker-V2/tree/main)  
```
├── ComfyUI/models/
|      ├──photomaker/
|             ├── photomaker-v1.bin
|             ├── photomaker-v2.bin
```

3.1.2 using dual role same frame function:      
Need download "ms_adapter.bin" : [link](https://huggingface.co/doge1516/MS-Diffusion/tree/main)    
Need clip_vision model "clip_g.safetensors" or other base from "CLIP-ViT-bigG-14-laion2B-39B-b160k";   

```
├── ComfyUI/models/
|      ├──photomaker/
|             ├── ms_adapter.bin
|      ├──clip_vision/
|             ├── clip_vision_g.safetensors(2.35G) or CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors(3.43G)
```

3.2 if using kolors:  
Kwai-Kolors    [link](https://huggingface.co/Kwai-Kolors/Kolors/tree/main)    
Kolors-IP-Adapter-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/tree/main)   
The file structure is shown in the following figure:
```
├── any path/Kwai-Kolors/Kolors
|      ├──model_index.json
|      ├──vae
|          ├── config.json
|          ├── diffusion_pytorch_model.safetensors (rename from diffusion_pytorch_model.fp16.safetensors )
|      ├──unet
|          ├── config.json
|          ├── diffusion_pytorch_model.safetensors (rename from diffusion_pytorch_model.fp16.safetensors )
|      ├──tokenizer
|          ├── tokenization_chatglm.py
|          ├── tokenizer.model
|          ├── tokenizer_config.json
|          ├── vocab.txt text_encoder
|       ├── text_encoder
|          ├── config.json
|          ├── configuration_chatglm.py
|          ├── modeling_chatglm.py
|          ├── pytorch_model.bin.index.json
|          ├── quantization.py
|          ├── tokenization_chatglm.py
|          ├── tokenizer.model
|          ├── tokenizer_config.json
|          ├── vocab.txt
|          ├── pytorch_model-00001-of-00007.bin to pytorch_model-00007-of-00007.bin
|       ├── scheduler
|          ├── scheduler_config.json
|       ├── Kolors-IP-Adapter-Plus
|          ├──model_index.json
|          ├──ip_adapter_plus_general.bin
|          ├──config.json
|          ├──image_encoder
|               ├──config.json
|               ├──preprocessor_config.json
|               ├──pytorch_model.bin
|               ├──tokenizer.json
|               ├──tokenizer_config.json
|               ├──vocab.json
```

3.3 The model file example for dual role controllnet is as follows, which only supports SDXL community controllnet    
```
├── ComfyUI/models/controlne/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|     ├──controlnet-zoe-depth-sdxl-1.0  
|     ├──TheMistoAI/MistoLine 
|     ├──xinsir/controlnet-tile-sdxl-1.0

```
Control_img image preprocessing, please use other nodes     

4 Example
----
txt2img mode use NF4 FLUX (Latest version)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/nf4.png)

img2img mode use auraface photomake V2  (Latest version)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/newest.png)

txt2img mode uses kolors model using chinese prompt (Outdated version examples)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/new.png)

img2img mode, uses kolors model using chinese prompt (Outdated version examples)     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/kolorsimg2img.png)

img2img mode, uses photomakeV1 (Outdated version examples)     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/newimg2imgV1.png)

img2img mode, uses photomakeV2 (Outdated version examples)     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/newimg2imgV2.png)

flux model,repo+pt txt2img/img2img(Outdated version examples)     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux_transfomerpy.png)

txt2img2role in 1 image (Outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/newtxt2img2role.png)

img2img2role in 1 image (Outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/2rolein1img.png)

img2img2role in 1 image lora (Outdated version examples)    
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/2rolein1imglora.png)

ControlNet added dual role co frame (Role 1 and Role 2) (Outdated version examples)  
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/controlnet.png)

Translate the text into other language examples, and the translation nodes in the diagram can be replaced with any translation node. Outdated version examples     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/trans1.png)

Function Description of Nodes  
---   
**<Storydiffusion_Model_Loader>**    
* repo: using diffuser models ;     
* ckptname:  using  community SDLX model selection;   
* vae_id: some model need fb16 vae,
* character_weights: Character weights saved using the save_character feature of the sampler node. Selecting "none/none" does not take effect! (Note that the saved character weights cannot be immediately recognized and require a restart of comfyUI);   
* lora: Selecting SDXL Lora does not take effect when set to "none";   
* lora_scale: The weight of Lora, which is enabled when Lora takes effect;   
* trigger_words: The keyword for Lora will be automatically added to the prompt. When enabling Lora, please fill in the corresponding trigger_words for Lora;   
* Scheduler: When selecting a sampler and adding characters to the same frame in the text and animation, if running continuously, an error will be reported. At this time, changing a sampler can continue running, but this bug has not been fixed yet;   
* model_type: Select either the txt2img or img2img mode, and when using the txt2img mode, the sampler may not be connected to the image;   
* id_number: How many roles are used, currently only supporting 1 or 2;   
* sa32_degree/sa64_degree: an adjustable parameter for the attention layer;   
* img_width/img_height: The height and width dimensions of the drawing.   
* photomake_mode: choice v1 or v2 model;  
* easy_function: try some new function... 

**<Storydiffusion_Sampler>**      
* pipe/info: The interface that must be linked;   
* image: The interface that must be linked to the image generation diagram. For dual roles, please follow the example and use the built-in image batch node in comfyUI;   
* character prompt: The prompt for the character, [character name] must be at the beginning. If using the graphic mode, the keyword "img" must be added, such as a man img;if using  chinese prompt, need["角色名"] or ['角色名']  
* Scene prompts: The prompt for the scene description, [character name], must start at the beginning. It is best for both characters to appear once in the first two lines. [NC] At the beginning, the character does not appear (suitable for non character scenes). When [character A] and [character B], MS diffusion's dual character mode is enabled, and and the spaces before and after it cannot be ignored# Used for segmented prompt, rendering the entire segment, but only outputting the prompt after #;    
* split prompt: The symbol for splitting the prompt, which does not take effect when it is empty. It is used to normalize paragraphs when the prompt is external. For example, when you pass in 10 lines of text, the hyphen may not be correct, but using a hyphen, such as ";", can effectively distinguish each line.     
* negative prompt: only effective when img_style is No_style;      
* seed/steps/cfg: suitable for commonly used functions in comfyUI;     
* ip-adapter_strength: img2img controls the weight of ip-adapter in graph generation,only using in kolors;   
* style_strength'ratio: Style weight control, which controls from which step the style takes effect. When the style consistency is not good, you can try increasing or decreasing this parameter;   
* clip_vison: Only valid when two characters are in the same image ,need "clip_g.safetensors" in models/clip_vison;   
* role-scale: only effective when two characters are in the same image, controlling the weight of the characters in the image;   
* Mask_threshold: It is only effective when two roles are in the same picture, and controls the position of the role in the picture (MS system automatically assigns the role position according to prompt, so appropriate role position information description can be added to prompt);   
* Start_step: Only effective when two characters are in the same image, controlling the number of starting steps for the character's position in the image   
* Save_character: Whether to save the character weights of the current character, file in/ Under ComfyUI_StoryDiffusion/weights/pt, use time as the file name;  
* Controlnet_modelpath: Controlnet's community  model );   
* Controllet_scale: control net weight;   
* guidance_list: contrlol role's position;     

**<Comic_Type>**        
* Fonts list: The puzzle node supports custom fonts (place the font file in the fonts directory. fonts/you_font. ttf);   
* Text_size: The size of the puzzle text;   
* Comic_type: Display the style of the puzzle;   
* Split lines: Suitable for non English text that has been translated by other translation nodes and the line break is removed. In this case, using a split symbol can correctly reassign the prompt line break to ensure that the text description is displayed on the correct image;   

**<Pre_Translate_prompt>: Pre processing of translation nodes**      
* Keep_charactername: Whether to keep the character name displayed on subsequent text puzzles.   

**Tips：**
* Add dual character same frame function, usage method: [A] .. [B]..., A. B is the role name, The parentheses are the effective conditions!!!   
* Optimize the loading of Lora's code, and when using accelerated Lora, trigger_words will no longer be added to the prompt list;   
* Playground v2.5 can be effective on txt2img, and there is no Playground v2.5 style Lora available when accelerated Lora can be used;   
* Role-scale, mask_threshold, and start_step mainly regulate the randomness and style consistency of two characters in the same frame;   
* The consistency of style can be adjusted between ip-adapter_strength and style_strength'ratio in img2img;   
* Preprocess translation text nodes, please refer to the example diagram for usage methods. (Pay attention to changing the font for Chinese or other East Asian characters);    * By default, use the ";" at the end of each paragraph to divide the paragraph. After translation into Chinese, there is a chance that it will be translated as ";", so remember to change it to ";", otherwise it will be a sentence.   
* The process of generating images using PhotosMaker requires the IMG keyword in the character prompt column. You can use keywords such as a woman IMG, a man IMG, etc;   
* No characters appear in the image, add [NC] in front of the scene prompt;   
* Segmented prompt, using #, such as AAAA # BBBB, will generate AAAA content, but the text will only display BBBB   
  
My ComfyUI node list：
-----

1、ParlerTTS :[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B :[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion ：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator ： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion ：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops ：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0  ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 ：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet ：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story  :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)       
12、ComfyUI_Streamv2v_Plus :[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
13、ComfyUI_MS_Diffusion :[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
14、ComfyUI_AnyDoor : [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
15、ComfyUI_Stable_Makeup : [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
16、ComfyUI_EchoMimic :  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
17、ComfyUI_FollowYourEmoji : [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
18、ComfyUI_Diffree : [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)     
19、ComfyUI_FoleyCrafter : [ComfyUI_FoleyCrafter](https://github.com/smthemex/ComfyUI_FoleyCrafter)
20、ComfyUI_MooER : [ComfyUI_MooER](https://github.com/smthemex/ComfyUI_MooER)

Citation
------

StoryDiffusion
``` python  
@article{zhou2024storydiffusion,
  title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
  author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
  journal={arXiv preprint arXiv:2405.01434},
  year={2024}
}
```
IP-Adapter
```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```
MS-Diffusion
```
@misc{wang2024msdiffusion,
  title={MS-Diffusion: Multi-subject Zero-shot Image Personalization with Layout Guidance}, 
  author={X. Wang and Siming Fu and Qihan Huang and Wanggui He and Hao Jiang},
  year={2024},
  eprint={2406.07209},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
photomaker
```
@inproceedings{li2023photomaker,
  title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
  author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
kolors
```
@article{kolors,
  title={Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis},
  author={Kolors Team},
  journal={arXiv preprint},
  year={2024}
}
```
