# ComfyUI_StoryDiffusion
本节点主要方法来源于StoryDiffusion，部分内容也来源于MS-Diffusion，感谢他们的开源！   
StoryDiffusion方法的地址: [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion)  以及 MS-Diffusion的地址: [link](https://github.com/MS-Diffusion/MS-Diffusion)

## 更新:
**2024/09/15**  
* 中秋节快乐！！
* 加入flux pulid 支持，目前fp8，和fp16能正常出图，但是fp16需要30G以上显存，可以忽略，需要有flux的diffuser文件(在repo输入)，以及对应的模型，然后easy function 填入pilid,fp8,cpu就可以开启，如果你的显存大于16G可以试试取消cpu，这样会快一点。nf4也能跑通，但是量化的思路不同，无法正常出图
* 加入kolor face id的支持，开启条件，在easyfunction里输入face，然后repo输入你的kolor diffuser模型的绝对路径地址。

**既往更新**  
* 加入diffuser尚未PR的图生图代码，fp8和fn4都能跑，还是nf4吧，快很多。图生图的噪声控制，由ip_adapter_strength的参数控制，越大噪声越多，当然图片不像原图，反之亦然。然后生成的实际步数是 你输入的步数*ip_adapter_strength的参数，也就是说，你输入50步，strength是0.8，实际只会跑40步。  
* 双角色因为方法的原因无法使用非正方形图片，所以用了讨巧的方法，先裁切成方形，然后再裁切回来；
* 高宽的全局变量名会导致一些啼笑皆非的错误，所以改成特别点的；
* 现在如果只使用flux的repo模型，不再自动保存一个pt文件，除非你在easy function输入save；
* 使用SDXL单体模型时，可能会报错，是因为runway 删除了他们的模型库，所以加入了内置的config文件，避免加载出错，这样的另一个好处，就是首次使用时，不用连外网了。
* 加载nf4模型的速度比fp8快了许多倍，所以我推荐使用nf4模型来运行flux。我已经把nf4的工作流放入example，只需下载单体模型地址，[link](https://huggingface.co/sayakpaul/flux.1-dev-nf4/tree/main) ，当然flux的完整diffuser模型也是必须的。  
* 加入easy function，便于调试新的功能，此次加入的是photomake V2对auraface的支持，你可以在easy function 输入auraface以测试该方法   
* 如果单独运行flux的repo，会自动保存pt模型（fp8)的，你可以运行至模型保存后就中断，然后用repo+pt模型，或者repo+其他fp8模型，或者repo+重新命名的pt模型（不带transformer字眼即可）来使用flux，速度更快。单独加载repo很耗时。   
* 特别更新：现在双角色同框的加载方式改成[A]...[B]...模式，原来的（A and B）模式已经摈弃摒弃！！！！   
* 特别注意，因为可灵模型比较大，所以采用了CPU加载，所以首次加载需要很大的内存才行。   
* 加入可灵kolor模型的支持，支持文生图和可灵ipadapter的图生图，需要的模型文件见下方；   
* 加入photomakerV2的支持，由于V2版需要insight face ，所以不会装的谨慎尝试；     
* controlnet现在使用单体模型；  
* 修复引入模块的错误，现在模型存放地址改至models/photomaker，重复利用模型，避免浪费硬盘空间(存储的pt模型也会在photomaker下)；   
* 新增controlnet布局控制按钮，默认是否，为程序自动。   
* 为双角色同图引入controlnet，并支持多图引入（MS还是保留吧，剔除了有些人又不想装2个插件。 ）  
* 加入角色模型保存和加载功能      

1.安装
-----
  在/ComfyUI /custom_node的目录下：   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git
  
  ```
或者用manager 安装。。   
 
2.需求文件   
----
```
pip install -r requirements.txt
```
如果要使用photomake v2
```
pip install insightface==0.7.3   
```
如果缺失模块，请单独pip install    

 
3 Need  model 
----
3.1 在线模式   
你可以直接在repo填写如：stabilityai/stable-diffusion-xl-base-1.0 ，也可以直接选择单体的SDXL社区模型。社区模型的优先级要高于repo模型。   
repo模式 支持所有基于SDXL的扩散模型（如G161222/RealVisXL_V4.0，sd-community/sdxl-flash），也支持非SD模型，如（stablediffusionapi/sdxl-unstable-diffusers-y，playground-v2.5-1024px-aesthetic）   
单体模型支持SDXL,例如：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors ），    

--(使用双角色功能时):       
你用全局外网，会自动下载。  
在comfyUI/models/photomaker目录下，确认是否有photomaker-v1.bin，如果没有会自己下载 [离线下载地址](https://huggingface.co/TencentARC/PhotoMaker/tree/main)  
photomaker-v2.bin 虽然也能用，但是新代码没有更新，所以发挥不了其新特性 [离线下载地址](https://huggingface.co/TencentARC/PhotoMaker-V2/tree/main)  

需要下载 "ms_adapter.bin" : [下载](https://huggingface.co/doge1516/MS-Diffusion/tree/main)   
需要下载 "clip_g.safetensors"或者任何其他基于 "CLIP-ViT-bigG-14-laion2B-39B-b160k"的单体模型,通常是4G左右，放在models/clip_vison目录下,  
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
Kolors-IP-Adapter-FaceID-Plus  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus)

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
如果使用kolor的face ip还需要:  
自动下载的insightface模型 "DIAMONIK7777/antelopev2" insightface models....
ipa-faceid-plus.bin 模型下载地址，放在如下目录  [link](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus)
```
├── ComfyUI/models/photomaker/
|             ├── ipa-faceid-plus.bin
```

3.2 离线模式 
可以在repo填写扩散模型的绝对地址，须用“/” .  

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

3.4 如果要使用 flux pulid  .   
确保torch must > 0.24.0，并确保optimum-quanto为0.2.4以上版本   
```
pip install optimum-quanto==0.2.4  
```
EVA02_CLIP_L_336_psz14_s6B.pt 会自动下载....[link](https://huggingface.co/QuanSun/EVA-CLIP/tree/main) #迟点改成不自动下载      
DIAMONIK7777/antelopev2 会自动下载，kolor也用这个....[https://huggingface.co/DIAMONIK7777/antelopev2/tree/main)    
"pulid_flux_v0.9.0.safetensors" 下载至 [link](https://huggingface.co/guozinan/PuLID/tree/main)     
fp8 using flux1-dev-fp8.safetensors  这个unt很多人应该有，放在checkpoints目录 [link](https://huggingface.co/Kijai/flux-fp8/tree/main)       
```
├── ComfyUI/models/photomaker/
|             ├── pulid_flux_v0.9.0.safetensors
```
确保 ae.safetensors 在你的 FLUX.1-dev 目录下,以下是文件夹示例:  
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


4 Example
----
请看英文版的提示

节点的功能说明
---   
**<Storydiffusion_Model_Loader>**  
*  repeo：填写扩散模型的绝对路径；   
*  ckpt_name：社区SDLX模型选择；   
*  vae_id:有些模型需要fb16的vae，你可以选择comfyUI的vae来避免出黑图
*  character_weights：使用sampler节点的save_character 功能保存的角色权重。选择为“none/无”时不生效！（注意，保存的角色权重不能马上被识别，需要重启comfyUI）；   
*  lora：选择SDXL lora，为“none”时不生效；   
*  lora_scale： lora的权重，Lora生效时启用；   
*  trigger_words： lora的关键词，会自动添加到prompt里，启用Lora时，请填写Lora对应的trigger_words；  
*  scheduler： 采样器选择，文生图加角色同框时，如果连续跑，会报错，这时候，改一个采样器，就能继续跑，这个bug暂时没修复；  
*  model_type： 选择txt2img 或者img2img模式，使用txt2img模式时采样器可以不接入图片；   
*  id_number： 使用多少个角色，目前仅支持1个或者2个；  
*  sa32_degree/sa64_degree： 注意力层的可调参数；  
*  img_width/img_height： 出图的高宽尺寸。
* photomake_mode： 选择用V1还是V2的模型；  
* easy_function  用来测试新功能的，目前输入auraface 会开启phtomake V2的auraface人脸识别功能，会下载模型，注意联网。   

**<Storydiffusion_Sampler>**
* pipe/info： 必须链接的接口；  
* image： 图生图才必须链接的接口，双角色请按示例，用comfyUI内置的image batch 节点；   
* character_prompt： 角色的prompt，[角色名] 必须在开头，如果使用图生图模式，必须加入“img”关键词，例如 a man img；    
* scene_prompts： 场景描述的prompt，[角色名] 必须在开头，2个角色最好在前两行各自出现一次，[NC]在开头时，角色不出现（适合无角色场景），(角色A and 角色B) 时开启MS-diffusion的双角色模式，and 和其前后空格不能忽略； #用于分段prompt，渲染整段，但是将只输出#后面的prompt；    
* split_prompt： 切分prompt的符号,为空时不生效，用于prompt为外置时的规范化段落。比如你传入10行的文字时，分段符不一定正确，但是用切分符号，比如“；”就能很好的区分每一行。    
* negative_prompt： 只在img_style为No_style时生效；     
* seed/steps/cfg： 适用于comfyUI常用功能；    
* ip_adapter_strength： kolor img2img 图生图的ip_adapter权重控制；  
* style_strength_ratio： 风格权重控制，控制风格从哪一步开始生效，风格一致性不好时，可以试着调高或者调低此参数；  
* clip_vison： 仅在双角色同图时有效，使用带g的clip_vision模型；    
* role_scale： 仅在双角色同图时有效，控制角色在图片中的权重；    
* mask_threshold： 仅在双角色同图时有效，控制角色在图片中的位置（MS系统自动根据prompt分配角色位置，所以prompt中可以加入适当的角色位置信息描写），为0时启用自动模式；   
* start_step： 仅在双角色同图时有效，控制角色在图片中的位置的起始步数;    
* save_character： 是否保存当前角色的角色权重，文件在./ComfyUI_StoryDiffusion/weigths/pt 下，以时间为文件名  ；  
* controlnet_model_path: 选择SDXL社区模型，仅双角色同框时有效；  
* controlnet_scale:  controlne权重；   
* guidance_list: 如果2组数字一样，人物的位置权重重叠，一般是用来分左右，上下，或其他方位；  

**<Comic_Type>** 
* fonts_list： 拼图节点支持自定义字体（把字体文件放在fonts目录下 .fonts/your_font.ttf）；
* text_size： 拼图文字的大小；    
* comic_type： 拼图的风格展示；  
* split_lines： 适用于非英语文字被其他翻译节点翻译后，换行符被删除，此时使用切分符号，可以正确地重新赋予prompt换行符，确保文字描述在正确的图片上显示；     

**<Pre_Translate_prompt> ： 翻译节点的前置处理**   
* keep_character_name： 是否保留角色名在后续文字拼图上显示。    

特别提醒：  

*  可灵中文输入，必须使用["角色名"]或者['角色名'],[NC]不变， 注意【】是不能用的！！！！  
*  可灵只支持在repo_id输入本地绝对地址，地址的最后部分必须是kolors   
*  可灵模型只需要下载fb16的，然后改名。
* 双角色同框功能，使用方法：[A]...  [B]....,有2个方括号在prompt中为生效条件！！！       
* 优化加载Lora的代码，使用加速Lora时，trigger_words不再加入prompt列表；    
* Playground v2.5可以在txt2img有效，没有Playground v2.5的风格Lora可用，当可以使用加速Lora;          
* role_scale，mask_threshold，start_step主要调节双角色同框的随机性和风格一致性；      
* ip_adapter_strength和style_strength_ratio在img2img时，可以调节风格的一致性；      
* 预处理翻译文本节点，使用方法可以参考示例图。  (中文或其他东亚文字注意更换字体)；         
* 默认用每段文字末尾的";"来切分段落，翻译为中文后，有几率会被翻译为“；”，所以记得改成“；”，否则会是一句话。              
* 图生图流程使用photomaker，角色prompt栏里，必须有img关键词，你可以使用a women img, a man img等；         
* 如果需要图片不出现角色，场景prompt前面加入[NC] ；     
* 分段prompt，用#，例如 AAAA#BBBB,将生成AAAA内容，但是文字只显示BBBB   

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
