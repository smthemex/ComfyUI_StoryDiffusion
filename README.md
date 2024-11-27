<h1> ComfyUI_StoryDiffusion：using StoryDiffusion in ComfyUI.</h1>

* [中文说明](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/README-CN.md)  
* StoryDiffusion origin From: [link](https://github.com/HVision-NKU/StoryDiffusion)
* The project also uses the following open-source projects:[MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion),[StoryMaker](https://github.com/RedAIGC/StoryMaker)，[consistory](https://github.com/NVlabs/consistory),[kolor](https://github.com/Kwai-Kolors/Kolors),[pulid](https://github.com/ToTheBeginning/PuLID),[flux](https://github.com/black-forest-labs/flux),[photomaker](https://github.com/TencentARC/PhotoMaker),[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) 

## Updates:
**2024/11/27**
* change some ms-diffusion'codes,if you use "[A] a (man) img " in role prompts,will get better of MS mode.
  
## Function introduction  
**story-diffusion**    
* Support img2img & txt2img，All you need to do is select an SDXL model in“ckpt_name”menu  to get started, support sdxl lora;  
* If using img2img,You can choose between the v1 or v2 version of PhotoMaker checkpoints(Automatic downloads);  phtomaker v2 need insightface and the models that go with it;

**ms-diffusion**    
* Enabled when 2 characters are required to appear in the same image, When [roleA] and [roleB] appear in a scene prompt at the same time, it will be automatically enabled;
* Of course the "ms_adapter.bin" model is also required, and the "clip_vision_g.safetensors"I is selected in the clip_vision;
* MS supports ControlNet, and a dual-role prompt requires a preprocessed ControlNet picture.

**consistory**    
* Enable when entering 'consi' in easy_function;   
* Only the sdxl model is needed, no additional models are needed;  

**story-maker**  
* "story-maker“ is similar to "story-diffusion", currently only supports img2img, which features the ability to migrate the character's clothing and supports dual characters with the same picture;
* To turn on this function, you need to enter 'maker' in easy-function; Then select an sdxl model and select the "clip_vision_H.safetensors" model in the clip-vision,The companion “mask.bin” model and“insightface"model are automatically downloaded;   
* If you enter 'maker,dual', the function of using 'story-diffusion'  in the front section and using 'story-maker' in the same picture for both characters will be enabled
* The method requires a mask, so “RMBG-1.4” is built-in, which can be downloaded automatically；

**kolor**  
* With kolor, you need to enter the local path of kolor in the repo_id, using '/' splitting the directory;You can use "clip-vit-large-patch14.safetensors", or the image encoder in the repo;
* Kolor supports img2img (IPadapeter and FaceID), txt2img,The matching model will be automatically downloaded, and the details can be found in the README model content; 
* Kolor supports prompt input in all Chinese characters, note that the character name needs to be changed to ['张三']；
* using kolor FaceId function, need fill easy function "face",

**Flux and PULID-FLUX**  
* Flux supports img2img and txt2img, and supports FP8 and NF4 (recommended) quantization models；To enable it, enter the local path of flux diffuser in 'repo_id' and select the corresponding model in 'ckpt-name';example fill "X:/xxx/xxx/black-forest-labs/FLUX.1-dev"; 
* PULID-FLUX needs to connect to the dual clip nodes of comfy in clip, and select 'EVA02_CLIP_L_336_psz14_s6B.pt ' in  clip-vision, Select a Flux FP8('Kijai/flux-fp8' and "Shakker-Labs/AWPortrait-FL") model (with flux in the name),'ae.safetensors' in vae menu, fill in 'pulid, fp8' in 'easy-function' ;The accompanying 'insightface' model will be automatically downloaded; 

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
if using photomaker V2，pulid-flux，kolor，story-make:  
```
pip install insightface==0.7.3  or new   
```
If the module is missing, please pip install   

3 Need  model 
----
**3.1 base 1:(choice repo_id or ckpt_name)**     
3.1.1 ckpt_name: for example: Jumpernaut XL_v9-RunDiffusionPhoto_v2. safetensors   
3.1.2 repo_id:"stablityai/table diffusion xl base-1.0" or local "x:/xx/table diffusion xl base-1.0" # support  playground-v2.5-1024px-aesthetic   

**3.2 base 2:**   
photomaker-v1.bin    [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   
photomaker-v2.bin    [link](https://huggingface.co/TencentARC/PhotoMaker-V2/tree/main)  
```
├── ComfyUI/models/photomaker/
|             ├── photomaker-v1.bin
|             ├── photomaker-v2.bin
```
**3.3 optional function**

**3.3.1 if using dual role same frame function(ms-diffuion)**:   
Need download "ms_adapter.bin" : [link](https://huggingface.co/doge1516/MS-Diffusion/tree/main)    
Need clip_vision model "clip_g.safetensors" or other base from "CLIP-ViT-bigG-14-laion2B-39B-b160k";   
```
├── ComfyUI/models/photomaker/
|             ├── ms_adapter.bin
├── ComfyUI/models/clip_vision/
|             ├── clip_vision_g.safetensors(2.35G) or CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors(3.43G)
```
if using controlnet in ms-diffusion(Control_img image preprocessing, please use other nodes     ); 
```
├── ComfyUI/models/controlnet/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|     ├──controlnet-zoe-depth-sdxl-1.0  
|     ├──TheMistoAI/MistoLine 
|     ├──xinsir/controlnet-tile-sdxl-1.0
```

**3.3.2 if using kolors:**     
Kwai-Kolors    [link](https://huggingface.co/Kwai-Kolors/Kolors/tree/main)    
Kolors-IP-Adapter-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/tree/main)   
```
├── ComfyUI/models/photomaker/
|             ├── ip_adapter_plus_general.bin
```
Kolors-IP-Adapter-FaceID-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus)
```
├── ComfyUI/models/photomaker/
|             ├── ipa-faceid-plus.bin
```
and if using  Kolors-IP-Adapter-FaceID-Plus:  
will auto download "DIAMONIK7777/antelopev2" insightface models....

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
```
if using monolithic model,choice a clip vision such as: "clip-vit-large-patch14.safetensors:   
```
├── ComfyUI/models/clip_vision/
|             ├── clip-vit-large-patch14.safetensors  # Kolors-IP-Adapter-Plus or Kolors-IP-Adapter-FaceID-Plus using same checkpoints. 
```
or using default file such as below:   
```
├── any path/Kwai-Kolors/Kolors/
|       ├──Kolors-IP-Adapter-Plus  # if using Kolors-IP-Adapter-Plus
|          ├──model_index.json
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

**3.3.3 if using flux**

**3.3.3.1 if using fp8 repo_id**     
fill local flux repo dir in repo_id..      
 
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
**3.3.3.2 if using fp8 repo_id+ckpt_name**   
fill local flux repo dir in repo_id and choice fp8 ckpt;  
```
├── ComfyUI/models/checkpoints/
|             ├── flux1-dev-fp8.safetensors
```
**3.3.3.3 if using nf4 repo_id+ckpt_name**   
fill local flux repo dir in repo_id and choice nf4 ckpt;  
downlaod nf4 model  [link](https://huggingface.co/sayakpaul/flux.1-dev-nf4/tree/main)
```
├── ComfyUI/models/checkpoints/
|             ├── rename nf4 ckpt
```

**3.3.3.4 using flux pulid,clip+ckpt_name**     .   
torch must > 0.24.0   
optimum-quanto must >=0.2.4 
```
pip install optimum-quanto==0.2.4  
```
EVA02_CLIP_L_336_psz14_s6B.pt auto downlaod....[link](https://huggingface.co/QuanSun/EVA-CLIP/tree/main)      
DIAMONIK7777/antelopev2 auto downlaod....[https://huggingface.co/DIAMONIK7777/antelopev2/tree/main)    
"pulid_flux_v0.9.0.safetensors" download from [link](https://huggingface.co/guozinan/PuLID/tree/main)     
fp8 using flux1-dev-fp8.safetensors  from [link](https://huggingface.co/Kijai/flux-fp8/tree/main)       
make sure ae.safetensors in you FLUX.1-dev dir,example:    

```
├── ComfyUI/models/photomaker/
|             ├── pulid_flux_v0.9.0.safetensors
├── ComfyUI/models/clip_vision/
|             ├── EVA02_CLIP_L_336_psz14_s6B.pt
├── ComfyUI/models/checkpoints/
|             ├── flux1-dev-fp8.safetensors
├── ComfyUI/models/clip/
|             ├── t5xxl_fp8_e4m3fn.safetensors
|             ├── clip_l.safetensors
```

**3.5 if using storymake..**    
mask.bin from  [link](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#可以自动下载   
buffalo_l from  [link](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#自动下载   
RMBG-1.4 from  [link](https://huggingface.co/briaai/RMBG-1.4/tree/main)#自动下载   
```
├── ComfyUI/models/photomaker/
|         ├── mask.bin
├── ComfyUI/models/clip_vision/
|             ├── clip_vision_H.safetensors  #2.4G base in laion/CLIP-ViT-H-14-laion2B-s32B-b79K
├── ComfyUI/models/buffalo_l/
|         ├── 1k3d68.onnx
|         ├── 2d106det.onnx
|         ├── det_10g.onnx
|         ├── genderage.onnx
|         ├── w600k_r50.onnx
```
4 Example
----
**consistory**
* whtn fill 'consi' in easyfunction is enable.. (Latest version)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/consitstory.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/consistorylora.png)

**sd35**
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/nf4_using_comfyUIclipandvae.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/nf4_using_comfyUIclipandvaeL.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/sd35nf4singlefile.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/sd35.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/sd35img2imgnf4.png)

**prompt_tag**
* using tag MiniCPM & CogFlorence 连环画可能会好点,如果加上controlnet   最新示例 (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/tag_mini.png)
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/tag.png)

**pulid-flux**  
* flux img2img Two examples 图生图,两种示例,非最新示例 (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux_pulid_new.png)

**comfyUI-normal**  
flux normal
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/cf_flux_txt2img.png)
sd1.5
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/cy_sd_txt2img.png)

**story-make**   

img2img using controlnet and 2roles in 1 img  纯storymaker生成，最新示例 (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/storymake_control.png)

img2img  纯storymaker生成，非最新示例 (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/maker2role.png)

**kolor-face**   
img2img kolor face，参数输入没变化，非最新示例  (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/kolor.png)

**flux-nf4**   
* txt2img mode use NF4 FLUX 开启flux nf4模式,速度最快，非最新示例 (outdated version examples)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/nf4.png)

**ms-diffusion**   
* img2img2role in 1 image，双角色同图，最新示例 (new version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/msdiffuion.png)
* ControlNet added dual role co frame (Role 1 and Role 2) (Outdated version examples)  
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/controlnet.png)

**story-diffusion**   
* photomake v2 in img2img normal 最基础的story流程，非最新示例 (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/phtomakev2.png)
* txt2img using lora and comic node (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img_lora_comic.png)
* Translate the text into other language examples, and the translation nodes in the diagram can be replaced with any translation node. Outdated version examples     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/trans1.png)

Function Description of Nodes  
---   
**<Storydiffusion_Model_Loader>**    
* image: optional,The interface that must be linked to the image generation diagram. For dual roles, please follow the example and use the built-in image batch node in comfyUI; 
* controlnet:only ms function support;  
* clip:Connect the two clip nodes of Comfyui（using flux-pulid）
* character prompt: The prompt for the character, [character name] must be at the beginning. If using the graphic mode, the keyword "img" must be added, such as a man img;if using  chinese prompt, need["角色名"] or ['角色名']  
* repo_id: using diffuser models ;     
* ckpt_name:  using  community SDLX model selection;   
* vae_id: some model need fb16 vae,keep none is fine,
* character_weights: Character weights saved using the save_character feature of the sampler node. Selecting "none/none" does not take effect! (Note that the saved character weights cannot be immediately recognized and require a restart of comfyUI);   
* lora: Selecting SDXL Lora does not take effect when set to "none";   
* lora_scale: The weight of Lora, which is enabled when Lora takes effect;
* clip_vison: ms diffusion,story_maker,pulid_flux need clip_vision models;
* controlnet_model_path: ms diffusion use only;    
* trigger_words: The keyword for Lora will be automatically added to the prompt. When enabling Lora, please fill in the corresponding trigger_words for Lora;   
* Scheduler: When selecting a sampler and adding characters to the same frame in the text and animation, if running continuously, an error will be reported. At this time, changing a sampler can continue running, but this bug has not been fixed yet;      
* sa32_degree/sa64_degree: an adjustable parameter for the attention layer;   
* img_width/img_height: The height and width dimensions of the drawing.   
* photomake_mode: choice v1 or v2 model;  
* easy_function: try some new function... 

**<Storydiffusion_Sampler>**      
* model: The interface that must be linked;   
* Scene prompts: The prompt for the scene description, [character name], must start at the beginning. It is best for both characters to appear once in the first two lines. [NC] At the beginning, the character does not appear (suitable for non character scenes). When [character A] and [character B], MS diffusion's dual character mode is enabled, and and the spaces before and after it cannot be ignored# Used for segmented prompt, rendering the entire segment, but only outputting the prompt after #;    
* negative prompt: only effective when img_style is No_style;      
* seed/steps/cfg: suitable for commonly used functions in comfyUI;     
* ip-adapter_strength: img2img controls the weight of ip-adapter in graph generation,only using in kolors;   
* style_strength'ratio: Style weight control, which controls from which step the style takes effect. When the style consistency is not good, you can try increasing or decreasing this parameter;   
* role-scale: only effective when two characters are in the same image, controlling the weight of the characters in the image;   
* Mask_threshold: It is only effective when two roles are in the same picture, and controls the position of the role in the picture (MS system automatically assigns the role position according to prompt, so appropriate role position information description can be added to prompt)(ms-diffusion only);   
* Start_step: Only effective when two characters are in the same image, controlling the number of starting steps for the character's position in the image   
* Save_character: Whether to save the character weights of the current character, file in/ Under ComfyUI_StoryDiffusion/weights/pt, use time as the file name;  
* Controllet_scale: control net weight,(ms-diffusion only);   
* guidance_list: contrlol role's position(ms-diffusion only);     

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
* Preprocess translation text nodes, please refer to the example diagram for usage methods. (Pay attention to changing the font for Chinese or other East Asian characters);    
* By default, use the ";" at the end of each paragraph to divide the paragraph. After translation into Chinese, there is a chance that it will be translated as ";", so remember to change it to ";", otherwise it will be a sentence.   
* The process of generating images using PhotosMaker requires the "img" keyword in the character prompt column. You can use keywords such as a woman "img" , a man img , etc;   
* No characters appear in the image, add [NC] in front of the scene prompt;   
* Segmented prompt, using #, such as AAAA # BBBB, will generate AAAA content, but the text will only display BBBB   

Previous updates
----
* Added support for 'consistory', you can enable this feature by typing 'consi' in easy_function ('cache' and 'inject' are two additional features, you can try with larger VRAM);
* The 'consistory' mode only supports single subjects, but you can also use (cat), (boy), or (hat) to create two subjects, such as entering:' a curve [girl] and wearing a (hat) 'in the character bar，Example images can be viewed；you can use lora when using consistory mode;  
* if use comfyUI sd3.5 clip and sd 3.5vae( from sd3.5 repo),can load single checkpoint(fp16,nf4 ) which can infer in nf4 mode.（need newest diffusers）
* add sd3.5 large support,can infer in normal or nf4 mode,nf4 mode has two chocie: fill in all local sd3.5 repo(need pip install -U diffusers)  or fill  local sd3.5 repo and chocie [nf4](https://huggingface.co/sayakpaul/sd35-large-nf4/tree/main/transformer) checkpoint([example](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/sd35nf4singlefile.png)). if use nf4 need fill in 'nf4' in easyfunction.   
* add easy_function_lite node,you can use img2tag instead of scene prompts. The current models used are  [pzc163/MiniCPMv2_6-prompt-generator](https://huggingface.co/pzc163/MiniCPMv2_6-prompt-generator) and [thwri/CogFlorence-2-Large-Freeze](https://huggingface.co/thwri/CogFlorence-2-Large-Freeze) . Using "thwri/CogFlorence-2-Large-Freeze" requires inputting "flor" in the lite node's easy_function,Temporarily run CUDA during the testing phase.
* Reproduce the ControlNet control of Story-maker .Now, control-img is only applicable to methods using ControlNet and porting Samper nodes;
* if using  ControlNet  in Story-maker,maybe OOM(VRAM<12G),For detailed content, please refer to the latest example image;
* if vram >30G using fp16,do not fill in fp8,and chocie  fp16 weights,  
* If using flux-pulid, Please run according to the two methods in my example(new.json);
* fix some bugs,now 12G VRA runing 1 img in cpu need 317.6s It's 10 times faster than the previous unoptimized version. For 24G VRAM users, please provide feedback on the time if it runs successfully so that I can optimize it;
* Entering the National Day holiday, so if there are any issues with this update, it will take a few days to reply; 
* Add comfyUI wrapper,now You can freely use ComfyUI's regular SDXL, SD1.5 FLUX..., and call the prompt format of this prompt(There's no special benefit, just convenience);
* fix some bug,and Cleared and organized some code.
* Any method inserted in the regular comb process, such as connecting Lora after the model, should be able to function properly and has not been tested;
* now fill in 'cpu' in easy-function ,will use CPU insightface...   
* After testing, only 'Kijai/flux-fp8' and "Shakker-Labs/AWPortrait-FL" fp8 can produce images normally in pulid-flux mode, while the other fp8 or nf4 checkpoints are noises;
* if using pulid-flux,No need to enter 'cpu' in easyfunction. Now choose CPU offloading based on your VRAM, with the dividing points being VR>30G, 18G<VR<30G, or VR<18G;
* If you don't use Comfyui's clip, you can continue to use the full repo-id to run the pulid-flux now; 
* Now if using Kolor's "ip-adapter" or "face ID", you can choose the monolithic model of clip_vision (such as :"clip-vit-large-patch14.safetensors") to load the image encoder. The change this brings is that Kolor's local directory can delete all files in the "clip-vit-large-patch14-336" and "Kolors-IP-Adapter-Plus" folders. Of course, because comfyUI defaults to clip image processing of "224" size , while Kolor defaults to 336 size, there will be a loss of accuracy and quality. Please refer to the image comparison in readme for details.
* Another change is that we now need to port the model of "ip_adapter_plus_general.bin" in "kolor-ipadapter" to the "comfyUI/models/photomaker" directory;  
* For the convenience of use, the layout of the node has been adjusted again,delete "id number"(base character lines now),delete " model_type"(base input image [img2img function] or not [txt2img function] now), 'clip_vision' , 'controlnet' , 'character prompt' ,'image','control_image' now in model loader node now.   
* Now,using story_maker or pulid_flux function ,need choice clip vision model (use 'clip_vision' menu);   
* add StoryMaker from From: [StoryMaker](https://github.com/RedAIGC/StoryMaker) to make dual role..or normal img2img,as detailed in the following text 3.5 ,fill "maker,dual" in easy function to using StoryMaker for dual role; 
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
Consistory
```
@article{tewel2024training,
  title={Training-free consistent text-to-image generation},
  author={Tewel, Yoad and Kaduri, Omri and Gal, Rinon and Kasten, Yoni and Wolf, Lior and Chechik, Gal and Atzmon, Yuval},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={4},
  pages={1--18},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

FLUX
![LICENSE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
