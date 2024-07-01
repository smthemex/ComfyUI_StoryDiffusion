# ComfyUI_StoryDiffusion
You can using StoryDiffusion in ComfyUI 

StoryDiffusion origin From: [link](https://github.com/HVision-NKU/StoryDiffusion)   
MS-Diffusion origin From: [link](https://github.com/MS-Diffusion/MS-Diffusion)
----
My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)   

NEW Update
---
--为双角色同图引入controlnet，支持多图引入，但是计划剥离MS-diffusion，2个方法在一起不好玩。 
--加入角色模型保存和加载功能      
--已知文生图加双角色同图时，只能跑一次，再次跑，需要改一下模型加载的采样器或者别的选项的bug，暂时没时间修复；   
--Introducing Controlnet for dual character co image, supporting multi image introduction, but planning to split MS diffusion, combining the two methods is not fun.  
--Add the function of saving and loading character models   
--It is known that when adding dual characters to the Wensheng diagram, it can only be run once. If you run it again, you need to fix the bug in the sampler or other options loaded on the model. There is currently no time to fix it;   

Notice（节点的菜单功能说明 Special Function Description of Nodes Menu）  
---   
--<Storydiffusion_Model_Loader>   
-- sd_type：选择“Use_Single_XL_Model”时，可以使用社区SDXL模型，其他选项均为扩散模型；   
-- ckpt_name：使用“Use_Single_XL_Model”时生效，社区SDLX模型选择；   
-- character_weights：使用sampler节点的save_character 功能保存的角色权重。选择为“none/无”时不生效！（注意，保存的角色权重不能马上被识别，需要重启comfyUI）；   
-- lora：选择SDXL lora，为“none”时不生效；   
-- lora_scale： lora的权重，Lora生效时启用；   
-- trigger_words： lora的关键词，会自动添加到prompt里，启用Lora时，请填写Lora对应的trigger_words；  
-- scheduler： 采样器选择，文生图加角色同框时，如果连续跑，会报错，这时候，改一个采样器，就能继续跑，这个bug暂时没修复；  
-- model_type： 选择txt2img 或者img2img模式，使用txt2img模式时采样器可以不接入图片；   
-- id_number： 使用多少个角色，目前仅支持1个或者2个；  
-- sa32_degree/sa64_degree： 注意力层的可调参数；  
--img_width/img_height： 出图的高宽尺寸。

--<Storydiffusion_Sampler>   
-- pipe/info： 必须链接的接口；  
--image： 图生图才必须链接的接口，双角色请按示例，用comfyUI内置的image batch 节点；   
--character_prompt： 角色的prompt，[角色名] 必须在开头，如果使用图生图模式，必须加入“img”关键词，例如 a man img；    
--scene_prompts： 场景描述的prompt，[角色名] 必须在开头，2个角色最好在前两行各自出现一次，[NC]在开头时，角色不出现（适合无角色场景），(角色A and 角色B) 时开启MS-diffusion的双角色模式，and 和其前后空格不能忽略； #用于分段prompt，渲染整段，但是将只输出#后面的prompt；    
--split_prompt： 切分prompt的符号,为空时不生效，用于prompt为外置时的规范化段落。比如你传入10行的文字时，分段符不一定正确，但是用切分符号，比如“；”就能很好的区分每一行。    
--negative_prompt： 只在img_style为No_style时生效；     
--seed/steps/cfg： 适用于comfyUI常用功能；    
--ip_adapter_strength： img2img 图生图的ip_adapter权重控制；  
--style_strength_ratio： 风格权重控制，控制风格从哪一步开始生效，风格一致性不好时，可以试着调高或者调低此参数；  
--encoder_repo： 仅在双角色同图时有效，如果要使用本地模型，务必使用X:/XXX/XXX/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k  必须是“/”；   
--role_scale： 仅在双角色同图时有效，控制角色在图片中的权重；    
--mask_threshold： 仅在双角色同图时有效，控制角色在图片中的位置（MS系统自动根据prompt分配角色位置，所以prompt中可以加入适当的角色位置信息描写）；   
-- start_step： 仅在双角色同图时有效，控制角色在图片中的位置的起始步数;    
--save_character： 是否保存当前角色的角色权重，文件在./ComfyUI_StoryDiffusion/weigths/pt 下，以时间为文件名  ；  

--<Comic_Type>    
--fonts_list： 拼图节点支持自定义字体（把字体文件放在fonts目录下 .fonts/your_font.ttf）；
--text_size： 拼图文字的大小；    
--comic_type： 拼图的风格展示；  
--split_lines： 适用于非英语文字被其他翻译节点翻译后，换行符被删除，此时使用切分符号，可以正确地重新赋予prompt换行符，确保文字描述在正确的图片上显示；     

--<Pre_Translate_prompt> ： 翻译节点的前置处理   
--keep_character_name： 是否保留角色名在后续文字拼图上显示。    

--<Storydiffusion_Model_Loader>      
--Sd_type: When selecting "UseSingle_XL-Model", the community SDXL model can be used, and all other options are diffusion models;   
--Ckptname: Effective when using "UsesSingle_XL-Model", community SDLX model selection;   
--Character_weights: Character weights saved using the save_character feature of the sampler node. Selecting "none/none" does not take effect! (Note that the saved character weights cannot be immediately recognized and require a restart of comfyUI);   
--Lora: Selecting SDXL Lora does not take effect when set to "none";   
--Lora_scale: The weight of Lora, which is enabled when Lora takes effect;   
--Trigger_words: The keyword for Lora will be automatically added to the prompt. When enabling Lora, please fill in the corresponding trigger_words for Lora;   
--Scheduler: When selecting a sampler and adding characters to the same frame in the text and animation, if running continuously, an error will be reported. At this time, changing a sampler can continue running, but this bug has not been fixed yet;   
--Model_type: Select either the txt2img or img2img mode, and when using the txt2img mode, the sampler may not be connected to the image;   
--Idnumber: How many roles are used, currently only supporting 1 or 2;   
--Sa32_degree/sa64_degree: an adjustable parameter for the attention layer;   
--Img_width/img_height: The height and width dimensions of the drawing.   

--<Storydiffusion_Sampler>      
--Pipe/info: The interface that must be linked;   
--Image: The interface that must be linked to the image generation diagram. For dual roles, please follow the example and use the built-in image batch node in comfyUI;   
--Character prompt: The prompt for the character, [character name] must be at the beginning. If using the graphic mode, the keyword "img" must be added, such as a man img;   
--Scene prompts: The prompt for the scene description, [character name], must start at the beginning. It is best for both characters to appear once in the first two lines. [NC] At the beginning, the character does not appear (suitable for non character scenes). When (character A and character B), MS diffusion's dual character mode is enabled, and and the spaces before and after it cannot be ignored# Used for segmented prompt, rendering the entire segment, but only outputting the prompt after #;    
--Split prompt: The symbol for splitting the prompt, which does not take effect when it is empty. It is used to normalize paragraphs when the prompt is external. For example, when you pass in 10 lines of text, the hyphen may not be correct, but using a hyphen, such as ";", can effectively distinguish each line.     
--Negative prompt: only effective when img_style is No_style;      
--Seed/steps/cfg: suitable for commonly used functions in comfyUI;     
--Ip-adapter_strength: img2img controls the weight of ip-adapter in graph generation;   
--Style_strength'ratio: Style weight control, which controls from which step the style takes effect. When the style consistency is not good, you can try increasing or decreasing this parameter;   
--Encoder'repo: Only valid when two characters are in the same image. If you want to use a local model, be sure to use X:/XXX/XXX/laion/CLIP ViT bigG-14-laion2B-39B-b160k, which must be "/";   
--Role-scale: only effective when two characters are in the same image, controlling the weight of the characters in the image;   
--Mask_threshold: It is only effective when two roles are in the same picture, and controls the position of the role in the picture (MS system automatically assigns the role position according to prompt, so appropriate role position information description can be added to prompt);   
--Start_step: Only effective when two characters are in the same image, controlling the number of starting steps for the character's position in the image   
--Save_character: Whether to save the character weights of the current character, file in/ Under ComfyUI_StoryDiffusion/weights/pt, use time as the file name;   

--<Comic_Type>      
--Fonts list: The puzzle node supports custom fonts (place the font file in the fonts directory. fonts/you_font. ttf);   
--Text_size: The size of the puzzle text;   
--Comic_type: Display the style of the puzzle;   
--Split lines: Suitable for non English text that has been translated by other translation nodes and the line break is removed. In this case, using a split symbol can correctly reassign the prompt line break to ensure that the text description is displayed on the correct image;   

--<Pre_Translate_prompt>: Pre processing of translation nodes      
--Keep_charactername: Whether to keep the character name displayed on subsequent text puzzles.   

Tips 提醒：  

--添加双角色同框功能，使用方法：(A and B) have lunch...., A,B为角色名，中间的 and 和括号不能删除,括号为生效条件！！！     
--因为调用了MS-diffusion的功能，所以要使用双角色同框，必须添加encoder模型（laion/CLIP-ViT-bigG-14-laion2B-39B-b160k,无法替换为其他的）和ip-adapeter微调模型（ms_adapter.bin,无法替换）；    
--优化加载Lora的代码，使用加速Lora时，trigger_words不再加入prompt列表；    
--Playground v2.5可以在txt2img有效，没有Playground v2.5的风格Lora可用，当可以使用加速Lora;          
--role_scale，mask_threshold，start_step主要调节双角色同框的随机性和风格一致性；      
--ip_adapter_strength和style_strength_ratio在img2img时，可以调节风格的一致性；      
--预处理翻译文本节点，使用方法可以参考示例图。  (中文或其他东亚文字注意更换字体)；         
--默认用每段文字末尾的";"来切分段落，翻译为中文后，有几率会被翻译为“；”，所以记得改成“；”，否则会是一句话。    
--编辑config/models.yaml文件，记住用同样的格式，可以加入你喜欢的基于SDXL的扩散模型。         
--支持diffuser 0.28以上版本；         
--图生图流程使用photomaker，角色prompt栏里，必须有img关键词，你可以使用a women img, a man img等；         
--图片不出现角色，场景prompt前面加入[NC] ；     
--分段prompt，用#，例如 AAAA#BBBB,将生成AAAA内容，但是文字只显示BBBB   

--Add dual character same frame function, usage method: (A and B) have lunch, A. B is the role name, and the middle and parentheses cannot be removed. The parentheses are the effective conditions!!!   
--Because the MS diffusion function was called, in order to use dual role same frame, it is necessary to add an encoder model (laion/CLIP ViT bigG-14 laion2B-39B-b160k, which cannot be replaced with others) and an ip adapet fine-tuning model (ms-adapter.bin, which cannot be replaced);   
--Optimize the loading of Lora's code, and when using accelerated Lora, trigger_words will no longer be added to the prompt list;   
--Playground v2.5 can be effective on txt2img, and there is no Playground v2.5 style Lora available when accelerated Lora can be used;   
--Role-scale, mask_threshold, and start_step mainly regulate the randomness and style consistency of two characters in the same frame;   
--The consistency of style can be adjusted between ip-adapter_strength and style_strength'ratio in img2img;   
--Preprocess translation text nodes, please refer to the example diagram for usage methods. (Pay attention to changing the font for Chinese or other East Asian characters);    
--By default, use the ";" at the end of each paragraph to divide the paragraph. After translation into Chinese, there is a chance that it will be translated as ";", so remember to change it to ";", otherwise it will be a sentence.   
--Edit the config/models. yaml file and remember to use the same format to include your favorite SDXL based diffusion model.   
--Supports diffuser versions 0.28 and above;   
--The process of generating images using PhotosMaker requires the IMG keyword in the character prompt column. You can use keywords such as a woman IMG, a man IMG, etc;   
--No characters appear in the image, add [NC] in front of the scene prompt;   
--Segmented prompt, using #, such as AAAA # BBBB, will generate AAAA content, but the text will only display BBBB   
  


1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git
  
  ```

2.requirements  
----
```
pip install -r requirements.txt
```
```
pip install git+https://github.com/tencent-ailab/IP-Adapter.git   
```
如果缺失模块，请单独pip install    
If the module is missing, please pip install   

2.1如果使用的是comfyUI整合包 提示ip_adapter库找不到，可以尝试以下方法：  

在整合包的python_embeded目录下，复制插件的requirements.txt文件到这个目录，去掉这个文件里ip-adapter前面的#号，保存，再打开CMD，然后运行pip install -r requirements.txt --target= "你的路径/python_embeded/Lib/site-packages"   ，只把你的路径改成你的实际路径，其他不要动   

例如：pip install -r requirements.txt --target= "你的路径/python_embeded/Lib/site-packages"

2.2 如果实在装不上，你需要安装python3.10环境，git，以及pip，然后运行pip install git+https://github.com/tencent-ailab/IP-Adapter.git  安装好之后，在你独立安装的python目录下，找到lib/site-packages下的ip-Adapter文件夹，复制到你的/python_embeded/Lib/site-packages里去  

2.3秋叶包，理论上是按2.1的方法，如果不行，可以试试，2.2

 
3 Need  model 
----
3.1  online  在线模式   
点击运行，会自动从huggingface 下载所需模型，请确保你的的网络通畅，默认可用的模型有G161222/RealVisXL_V4.0 ，stabilityai/stable-diffusion-xl-base-1.0   ， stablediffusionapi/sdxl-unstable-diffusers-y ，sd-community/sdxl-flash ；  
选择'Use_Single_XL_Model',以及你本地的SDXL单体模型（例如：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors ），也会下载对应的config文件；  

In online mode, click run and the required model will be automatically downloaded from the huggingface. Please ensure that your network is unobstructed. The default available models are G161222/RealVisXL_V4.0, stabilityai/stable-diffusion-xl-base-1.0  ， stablediffusionapi/sdxl-unstable-diffusers-y ，sd-community/sdxl-flash ；    
Select 'Use_Single_XL-Model', as well as your local SDXL monomer model (for example: Jumpernaut XL_v9-RunDiffusionPhoto_v2. safetensors), and the corresponding config file will also be downloaded;    

--using dual role same frame function(使用双角色功能时):      

Need download "ms_adapter.bin" : [link](https://huggingface.co/doge1516/MS-Diffusion/tree/main) 
Need encoder model "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k":[link](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 

```
├── ComfyUI/custom_nodes/ComfyUI_StoryDiffusion/
|      ├──weights/
|             ├── photomaker-v1.bin
|             ├── ms_adapter.bin

```

3.2 offline  
打开ComfyUI_StoryDiffusion/config/models.yaml的models.yaml文件，如果有预下载的默认的扩散模型，可以不填，如果地址不在默认的C盘一类，可以在“path”一栏：填写扩散模型的绝对地址，须是“/” .  
Open the models.yaml file of ComfyUI_StoryDiffusion/config/models.yaml. If there is a pre downloaded default diffusion model, it can be left blank. If the address is not in the default C drive category, you can fill in the absolute address of the diffusion model in the "path" column, which must be "/"   

--using dual role same frame function(使用双角色功能时):     
在“laion/CLIP-ViT-bigG-14-laion2B-39B-b160k” 一栏里填写你的本地clip模型的绝对路径，使用“/”，需求的文件看下面的文件结构演示。      
Fill in the absolute path of your local clip model in the "laion/CLIP ViT bigG-14-laion2B-39B-b160k" column, using "/". Please refer to the file structure demonstration below for the required files.        
```
├── ComfyUI/custom_nodes/ComfyUI_StoryDiffusion/
|      ├──weights/
|             ├── photomaker-v1.bin
|             ├── ms_adapter.bin
├── Any local_path/
|     ├──CLIP ViT bigG-14-laion2B-39B-b160k/
|             ├── config.json
|             ├── preprocessor_config.json
|             ├──pytorch_model.bin.index.json
|             ├──pytorch_model-00001-of-00002.bin
|             ├──pytorch_model-00002-of-00002.bin
|             ├──special_tokens_map.json
|             ├──tokenizer.json
|             ├──tokenizer_config.json
|             ├──vocab.json
```

3.3 
在comfyUI的models目录下，确认是否有photomaker 目录，没有会自己新建并下载 photomaker-v1.bin   [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   
如果有预下载，就把模型放进去。  

make sure ..models/photomaker/photomaker-v1.bin    [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)     

3.4 双角色controlnet的模型文件示例如下，仅支持SDXL controlnet(The model file example for dual role controllnet is as follows, which only supports SDXL controllnet) 
```
├── ComfyUI/models/diffusers/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──TheMistoAI/MistoLine 
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors    
```
control_img图片的预处理，请使用其他节点(Control_img image preprocessing, please use other nodes)  

4 Example
----
txt2img and 2role in 1img  文生图双角色同框加双controlnet最新的流程
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img2controlnetimg.png)

![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2imgcontronet.png)

img2img 图生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2img.png)


txt2img lora and Dual role same fram  双角色同框并加入Lora，文生图示例
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2imglora2role.png)

1img2img and lora 图生图加风格lora
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/1imgtoimglora.png)

two character lighting lora  双角色及闪电lora  
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/2imgtoimglightinglora.png)

using other language    使用其他语言的文本,翻译节点请换成你有的。      
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/trans.png)



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
