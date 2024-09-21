# ComfyUI_StoryDiffusion
You can using StoryDiffusion in ComfyUI.

* [中文说明](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/README-CN.md)  
* StoryDiffusion origin From: [link](https://github.com/HVision-NKU/StoryDiffusion)  ---&---  MS-Diffusion origin From: [link](https://github.com/MS-Diffusion/MS-Diffusion)---&---StoryMakerr from From:[StoryMaker](https://github.com/RedAIGC/StoryMaker)


## Updates:
**2024/09/21**   
* add StoryMaker from From: [StoryMaker](https://github.com/RedAIGC/StoryMaker) to make dual role..or normal img2img,as detailed in the following text 3.5 ,fill "maker,dual" in easy function to using StoryMaker for dual role; 
  
**Previous updates：**  
* Add " PulID FLUX " function, In my testing, a minimum of 12GB of VRAM can run normally,now you can fill "X:/xxx/xxx/black-forest-labs/FLUX.1-dev",and fill easy function "pilid,fp8,cpu"(if you Vram>30G,can remove cpu,and using Kijai/flux-fp8,if you Vram>45G,can remove fp8,cpu), although it is a bit slow if using cpu! ,Of course, some models need to be prepared, as detailed in the following text;
* Add kolor FaceId function, now you can fill "xxx:/xxx/xxx/Kwai-Kolors/Kolors",and fill easy function "face",Of course, some models need to be prepared, as detailed in the following text; 
* Add diffusers'img2img codes( Not commit diffusers yet),Now you can using flux img2img function. in flux img2img,"guidance_scale" is usually 3.5 ,you can change ip-adapter_strength's number to Control the noise of the output image, the closer the number is to 1, the less it looks like the original image, and the closer it is to 0, the more it looks like the original image. Correspondingly, your generated step count is a multiple of this value, which means that if you enter 50 steps, the actual number of steps taken is 50 * 0.8 (0.8 is the value of change ip-adapter_strength) #you can see exmaple img
* AWPortrait-FL-fp8.safetensors is support if using fp8 mode,..  
* using img crop to fix ms_diffusion only using square's error;
* change W and H global names,it cause some error;
* fix runway error,when loader single model. 
* The loading speed of the NF4 model is many times faster than FP8, so I recommend using the NF4 model to run Flux. I have included the workflow of NF4 in the example，
* Add an "easy_function" for debugging new function. This time, I have added support for "auraface" in "Photomake V2". You can enter "auraface" into the "easy_function" to test this method
* support Flux ,1.only using fp8 repo,fill local "X:/xxx/xxx/black-forest-labs/FLUX.1-dev" and ckpt_name="none";if save .pt,fill easy function "save",2. using unet,fill local repo and choice a fp8 Unet(like Kijai/flux-fp8, AWPortrait-FL-fp8.safetensors. *.pt), 3. using fn4,need download weights at [link](https://huggingface.co/sayakpaul/flux.1-dev-nf4/tree/main),put weight in "comfyui/models/checkpoints/";fill local "X:/xxx/xxx/black-forest-labs/FLUX.1-dev"  
* Now clip checkpoints no need diffusers_repo,you can using "clip_g.safetensors" or other base from "CLIP-ViT-bigG-14-laion2B-39B-b160k";       
* 2 role in 1 img now using [A]...[B]... mode,  
* Support "kolors" text2img and "kolors"ipadapter img2img,using repo like :"xxx:/xxx/xxx/Kwai-Kolors/Kolors"  (Please refer to the end of the article for detailed file combinations)  
* support photomaker V2;  
* ControlNet now uses community models.   
* The base model now has only two options: using repo input or selecting the community model...  
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
3.1.1base:   
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
Kolors-IP-Adapter-FaceID-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus)

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
|       ├── Kolors-IP-Adapter-Plus  # if using Kolors-IP-Adapter-Plus
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
|       ├── clip-vit-large-patch14-336  # if using Kolors-IP-Adapter-FaceID-Plus
|          ├──config.json
|          ├──merges.txt
|          ├──preprocessor_config.json
|          ├──pytorch_model.bin
|          ├──special_tokens_map.json
|          ├──tokenizer.json
|          ├──tokenizer_config.json
|          ├──vocab.json
```
and if using  Kolors-IP-Adapter-FaceID-Plus:  
will auto download "DIAMONIK7777/antelopev2" insightface models....
ipa-faceid-plus.bin :Kolors-IP-Adapter-FaceID-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus)
```
├── ComfyUI/models/photomaker/
|             ├── ipa-faceid-plus.bin
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

3.4 using flux pulid  .   
torch must > 0.24.0   
```
pip install optimum-quanto==0.2.4  
```
EVA02_CLIP_L_336_psz14_s6B.pt auto downlaod....[link](https://huggingface.co/QuanSun/EVA-CLIP/tree/main) #迟点改成不自动下载      
DIAMONIK7777/antelopev2 auto downlaod....[https://huggingface.co/DIAMONIK7777/antelopev2/tree/main)    
"pulid_flux_v0.9.0.safetensors" download from [link](https://huggingface.co/guozinan/PuLID/tree/main)     
fp8 using flux1-dev-fp8.safetensors  from [link](https://huggingface.co/Kijai/flux-fp8/tree/main)       
```
├── ComfyUI/models/photomaker/
|             ├── pulid_flux_v0.9.0.safetensors
```
make sure ae.safetensors in you FLUX.1-dev dir,example:  
```
├──any_path/black-forest-labs/FLUX.1-dev
|      ├──model_index.json
|      ├──ae.safetensors
|      ├──vae
|          ├── config.json
|          ├── diffusion_pytorch_model.safetensors 
|      ├──transformer
|          ├── config.json
|          ├──diffusion_pytorch_model-00001-of-00003.safetensors
|          ├──diffusion_pytorch_model-00002-of-00003.safetensors
|          ├──diffusion_pytorch_model-00003-of-00003.safetensors
|          ├── diffusion_pytorch_model.safetensors.index.json
|      ├──tokenizer
|          ├── special_tokens_map.json
|          ├── tokenizer_config.json
|          ├── vocab.json
|          ├── merges.txt
|      ├──tokenizer_2
|          ├── special_tokens_map.json
|          ├── tokenizer_config.json
|          ├── spiece.model
|          ├── tokenizer.json
|       ├── text_encoder
|          ├── config.json
|          ├── model.safetensors
|       ├── text_encoder_2
|          ├── config.json
|          ├── model-00001-of-00002.safetensors
|          ├── model-00002-of-00002.safetensors
|          ├── model.safetensors.index.json
|       ├── scheduler
|          ├── scheduler_config.json
```
3.5 using storymake..    
mask.bin from  [link](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#可以自动下载   
buffalo_l from  [link](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#自动下载   
RMBG-1.4 from  [link](https://huggingface.co/briaai/RMBG-1.4/tree/main)#自动下载   
laion/CLIP-ViT-H-14-laion2B-s32B-b79K  from  [link](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)#自动下载   
```
├── ComfyUI/models/photomaker/
|         ├── mask.bin
├── ComfyUI/models/buffalo_l/
|         ├── 1k3d68.onnx
|         ├── 2d106det.onnx
|         ├── det_10g.onnx
|         ├── genderage.onnx
|         ├── w600k_r50.onnx
```
4 Example
----
img2img mode use storymaker for dual role(Latest version) 
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/using_storymake_dual_onlyA.png)

img2img mode use flux pulid  12G Vram,cpu(Latest version) 
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux_pulid_fp8_12GVR.png)

img2img mode use nf4 flux (Latest version)  
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux_img2img.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux_img2img2role.png)

txt2img mode use NF4 FLUX (Latest version)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/nf4.png)

img2img mode use auraface photomake V2  (Latest version)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/newest.png)

img2img model use kolors ip adapter,and using chinese prompt  (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/kolor_ipadapter_use_chinese.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/kolor_ipadapter.png)

img2img model use kolors ip adapter face id  (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/kolor_faceid.png)

img2img sdxl mode, uses photomakeV1 (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/photomakev1.png)

img2img sdxl  mode, uses photomakeV2 (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/photomakev2.png)

txt2img using lora and comic node (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img_lora_comic.png)

img2img2role in 1 image (Outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/2rolein1img.png)

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
PuLID
```
@article{guo2024pulid,
  title={PuLID: Pure and Lightning ID Customization via Contrastive Alignment},
  author={Guo, Zinan and Wu, Yanze and Chen, Zhuowei and Chen, Lang and He, Qian},
  journal={arXiv preprint arXiv:2404.16022},
  year={2024}
}
```

FLUX
![LICENSE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
