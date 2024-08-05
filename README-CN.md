# ComfyUI_StoryDiffusion
本节点主要方法来源于StoryDiffusion，部分内容也来源于MS-Diffusion，感谢他们的开源！

StoryDiffusion方法的地址: [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion)   
MS-Diffusion的地址: [link](https://github.com/MS-Diffusion/MS-Diffusion)
----

更新
---
2024/08/05更新   
--特别注意，因为可灵模型比较大，所以采用了CPU加载，所以首次加载需要很大的内存才行，32G内存以下谨慎测试，别问为什么，除非你用超大显存。   
--加入可灵kolor模型的支持，支持文生图和可灵ipadapter的图生图，需要的模型文件见下方；   
-- 加入photomakerV2的支持，由于V2版需要insight face ，所以不会装的谨慎尝试；     
--修复一些bug     

2024/07/26更新   
--模型现在只有使用repo输入或者选择社区模型两种方式，修复了一些bug；  
--controlnet现在使用单体模型；  
---调整MS的模型加载，速度更快了；  

20270709更新
--修复文生图使用MS-diffusion时无法连续跑的bug，需要开启加载模型节点的“reset_txt2img”为Ture；  
--修复引入模块的错误，现在模型存放地址改至models/photomaker，重复利用模型，避免浪费硬盘空间(存储的pt模型也会在photomaker下)；   
-- 更改选择模型的方式，现在可以更方便选择其他的扩散模型了；   

--新增controlnet布局控制按钮，默认是否，为程序自动。
--修复controlnet加载菜单的bug；   
--为双角色同图引入controlnet，并支持多图引入（MS还是保留吧，剔除了有些人又不想装2个插件。 ）  
--加入角色模型保存和加载功能      
--已知文生图加双角色同图时，只能跑一次，再次跑如果报错，只需切换一下模型加载的采样器或者别的选项的，这个bug暂时没时间修复；

1.安装
-----
  在/ComfyUI /custom_node的目录下：   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git
  
  ```
或者用manage 安装。。   
 
2.需求文件   
----
```
pip install -r requirements.txt
```
如果要使用photomake v2
```
pip install insightface==0.7.3   或者更高版本（未测试）
```
如果缺失模块，请单独pip install    

 
3 Need  model 
----
3.1 在线模式   
你可以直接在repo填写如：stabilityai/stable-diffusion-xl-base-1.0 ，也可以直接选择单体的SDXL社区模型。社区模型的优先级要repo模型。   
repo模式 支持所有基于SDXL的扩散模型（如G161222/RealVisXL_V4.0，sd-community/sdxl-flash），也支持非SD模型，如（stablediffusionapi/sdxl-unstable-diffusers-y，playground-v2.5-1024px-aesthetic）   
单体模型支持SDXL,例如：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors ），    

--(使用双角色功能时):       
你用全局外网，会自动下载，但是一般是去C盘。  
在comfyUI/models/photomaker目录下，确认是否有photomaker-v1.bin，如果没有会自己下载 [离线下载地址](https://huggingface.co/TencentARC/PhotoMaker/tree/main)  
photomaker-v2.bin 虽然也能用，但是新代码没有更新，所以发挥不了其新特性 [离线下载地址](https://huggingface.co/TencentARC/PhotoMaker-V2/tree/main)  

需要下载 "ms_adapter.bin" : [下载](https://huggingface.co/doge1516/MS-Diffusion/tree/main) 
需要下载 "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k":[下载地址](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 
文件存放的结构如下：  
```
├── ComfyUI/models/
|      ├──photomaker/
|             ├── photomaker-v1.bin
|             ├── photomaker-v2.bin
|             ├── ms_adapter.bin

```

如果要使用kolor（可灵），下载链接如下：
Kwai-Kolors    [link](https://huggingface.co/Kwai-Kolors/Kolors/tree/main)    
Kolors-IP-Adapter-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/tree/main)   
文件结构如下，注意是有层级的：
```
├── 你的本地任意地址/Kwai-Kolors/Kolors
|      ├──model_index.json
|      ├──vae
|          ├── config.json
|          ├── diffusion_pytorch_model.safetensors (从diffusion_pytorch_model.fp16.safetensors 改名而来)
|      ├──unet
|          ├── config.json
|          ├── diffusion_pytorch_model.safetensors (从diffusion_pytorch_model.fp16.safetensors 改名而来)
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
|          ├── pytorch_model-00001-of-00007.bin to pytorch_model-00007-of-00007.bin（7个模型，别下少了）
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

3.2 离线模式 
可以在repo填写扩散模型的绝对地址，须用“/” .  

--(使用双角色功能时):     
在“laion/CLIP-ViT-bigG-14-laion2B-39B-b160k” 一栏里填写你的本地clip模型的绝对路径，使用“/”，需求的文件看下面的文件结构演示。      

以下是双角色功能，离线版的模型文件结构：   
```
├── 任意地址/
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

3.3 双角色controlnet的模型示例如下，现在已支持社区SDXL单体模型。
```
├── ComfyUI/models/controlnet/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|     ├──/controlnet-zoe-depth-sdxl-1.0  
|     ├──TheMistoAI/MistoLine 
|     ├──xinsir/controlnet-tile-sdxl-1.0
   
```
control_img图片的预处理，请使用其他节点   

4 Example
----

文生图模式，使用可灵的中文提示词，最新示例，example内最新的json文件      
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2imgkolors.png)

图生图模式，使用可灵的中文提示词，最新示例，example内最新的json文件      
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2imgkolors.png)

图生图模式，使用photomakeV2，最新示例，
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2imgphotomakev2.png)

图生图模式,加入Lora，加入双角色同框（角色1 and 角色2），加入controlnet控制（controlnet只能控制双角色同框，旧的示例，只供参考      
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2imgcontrolnetdual.png)

文生图模式,加入HYper 加速Lora，加入双角色同框（角色1 and 角色2），加入controlnet控制（controlnet只能控制双角色同框）旧的示例  只供参考       
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img_hyperlora_contrlnet_2role1img.png)

多controlnet加入双角色同框（角色1 and 角色2）旧的示例，只供参考       
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/controlnetnum.png)

文本翻译为其他语言示例，图示中的翻译节点可以替换成任何翻译节点。旧的示例只供参考       
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/trans1.png)

节点的功能说明
---   
--<Storydiffusion_Model_Loader>  
-- repeo：填写扩散模型的绝对路径；   
-- ckpt_name：社区SDLX模型选择；   
-- vae_id:有些模型需要fb16的vae，你可以选择comfyUI的vae来避免出黑图
-- character_weights：使用sampler节点的save_character 功能保存的角色权重。选择为“none/无”时不生效！（注意，保存的角色权重不能马上被识别，需要重启comfyUI）；   
-- lora：选择SDXL lora，为“none”时不生效；   
-- lora_scale： lora的权重，Lora生效时启用；   
-- trigger_words： lora的关键词，会自动添加到prompt里，启用Lora时，请填写Lora对应的trigger_words；  
-- scheduler： 采样器选择，文生图加角色同框时，如果连续跑，会报错，这时候，改一个采样器，就能继续跑，这个bug暂时没修复；  
-- model_type： 选择txt2img 或者img2img模式，使用txt2img模式时采样器可以不接入图片；   
-- id_number： 使用多少个角色，目前仅支持1个或者2个；  
-- sa32_degree/sa64_degree： 注意力层的可调参数；  
--img_width/img_height： 出图的高宽尺寸。
--photomake_mode： 选择用V1还是V2的模型；  
--reset_txt2img  文生图模式的BUG目前只能用开启这个来修复.   

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
--controlnet_model_path: 选择SDXL社区模型；  
--controlnet_scale:  controlne权重；   
--layout_guidance: 是否开启自动布局（如果开启自动布局，prompt里最好有明显的位置信息，比如在左边，在哪。。。，比如上下等等）；  

--<Comic_Type>  
--fonts_list： 拼图节点支持自定义字体（把字体文件放在fonts目录下 .fonts/your_font.ttf）；
--text_size： 拼图文字的大小；    
--comic_type： 拼图的风格展示；  
--split_lines： 适用于非英语文字被其他翻译节点翻译后，换行符被删除，此时使用切分符号，可以正确地重新赋予prompt换行符，确保文字描述在正确的图片上显示；     

--<Pre_Translate_prompt> ： 翻译节点的前置处理   
--keep_character_name： 是否保留角色名在后续文字拼图上显示。    

特别提醒：  

-- 可灵中文输入，必须使用["角色名"]或者['角色名'],[NC]不变， 注意【】是不能用的！！！！  
-- 可灵只支持在repo_id输入本地绝对地址，地址的最后部分必须是kolors   
-- 可灵模型只需要下载fb16的，然后改名。

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


我的其他comfyUI插件：
-----

1、ParlerTTS node （ParlerTTS英文的音频节点）:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node（羊驼3的节点，也兼容了其他基于羊驼3的模型）:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node（高清放大节点）：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node（零样本单图制作视频）： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node（故事绘本节点）：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node（材质、融合类节点，基于pops方法）：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node（SD官方的音频节点的简单实现） ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node（基于智普AI的api节点，涵盖智普的本地大模型）：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node（基于腾讯的CustomNet做的角度控制节点）：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node（方便玩家调用镜像抱脸下载） :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node（基于模型的图像识别） :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node（生成式PBR贴图，即将上线）:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)   
13、ComfyUI_Streamv2v_Plus node（视频转绘，能用，未打磨）:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node（基于MS-diffusion做的故事话本）:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node(一键换衣插件): [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node(一键化妆): [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node(音频驱动动画):  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node(画面驱动表情包): [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [超强的一致性的文生图内绘](https://github.com/smthemex/ComfyUI_Diffree)     

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
