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
1、添加双角色同框功能，使用方法：(A and B) have lunch...., A,B为角色名，中间的 and 和括号不能删除,括号为生效条件！！！   
2、因为调用了MS-diffusion的功能，所以要使用双角色同框，必须添加encoder模型（laion/CLIP-ViT-bigG-14-laion2B-39B-b160k,无法替换为其他的）和ip-adapeter微调模型（ms_adapter.bin,无法替换）；  
3、更改lora的代码，现在lora_scale应该正常生效了。

1. Add dual role same frame function, usage method: (A and B) have lunch...., where A and B are role names, and the middle "and" parentheses cannot be removed，The parentheses represent the effective conditions!!!   
2. Because the" MS diffusion" function is called, in order to use dual role same frame, it is necessary to add an encoder model (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k,which cannot be replaced with others) and an ip adapet fine-tuning model (ms_adapter.bin,which cannot be replaced);   
3. Change Lora's code, "Lora_scale" should now take effect normally.   


Notice（节点的特殊功能说明 Special Function Description of Nodes）  
---   
1、预处理翻译文本节点，使用方法可以参考示例图。  (中文或其他东亚文字注意更换字体)；       
2、默认用每段文字末尾的";"来切分段落，翻译为中文后，有几率会被翻译为“；”，所以记得改成“；”，否则会是一句话。   
3、编辑config/models.yaml文件，记住用同样的格式，可以加入你喜欢的基于SDXL的扩散模型。  
4、示例的playground模型仅是测试，但是无法出图，请勿使用。  
5、拼图节点支持自定义字体（把字体文件放在fonts目录下 .fonts/your_font.ttf）和字体大小，增加了双角色的支持（使用comfyUI的batch image）；    
6、可以使用单体SDXL模型，方法是 选择“Use_Single_XL_Model”，然后在ckpt_name菜单选择你想使用的XL模型；
7、加入Lora的支持，lora菜单选择"none"时，Lora无效；“trigger_words”须填入你选择的Lora模型的对应的trigger_words，无须在prompt中加入，插件会自动在每行的末尾加入；
8、支持diffuser 0.28以上版本；      
9、图生图流程使用photomaker，角色prompt栏里，必须有img关键词，你可以使用a women img, a man img等；       
10、图片不出现角色，场景prompt前面加入[NC] ；  
11、分段prompt，用#，例如 AAAA#BBBB,将生成AAAA内容，但是文字只显示BBBB

1. Preprocess translation text nodes, please refer to the example diagram for usage methods. (Pay attention to changing the font for Chinese or other East Asian characters);   
2. By default, use the ";" at the end of each paragraph to divide the paragraph. After translation into Chinese, there is a chance that it will be translated as ";", so remember to change it to ";", otherwise it will be a sentence.   
3. Edit the config/models. yaml file and remember to use the same format to include your favorite SDXL based diffusion model.   
4. The playground model in the example is only for testing purposes, but cannot be illustrated. Please do not use it.   
5. The jigsaw puzzle node supports custom fonts (placing font files in the fonts directory".fonts/you_font. ttf") and font sizes, and adds support for dual characters (using batch images in comfyUI);    
6. You can use a single SDXL model by selecting "Use_Single_XL-Model" and then selecting the XL model you want to use from the ckptname menu;   
7. Add support for Lora. When selecting "none" in the Lora menu, Lora becomes invalid; "Trigger_words" must be filled in with the corresponding trigger_words for the Lora model you have selected, without the need to add them in the prompt. The plugin will automatically add them at the end of each line;   
8. Supports diffuser versions 0.28 and above;   
9. The process of generating images using PhotosMaker requires the"img" keyword in the character prompt column. You can use keywords such as a woman img, a man img, etc;    
10. No characters appear in the image, add [NC] in front of the scene prompt;   
11. Segmented prompt, using #, such as AAAA # BBBB, will generate AAAA content, but the text will only display BBBB   


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
├── ComfyUI/custom_nodes/ComfyUI_Pops/
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
├── ComfyUI/custom_nodes/ComfyUI_Pops/
|      ├──weights/
|             ├── photomaker-v1.bin
|             ├── ms_adapter.bin
├── local_path/
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

4 Example
----

Dual role same frame and lora  txt2img  双角色同框并加入Lora，文生图示例
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2imgloraand2char.png)

txt2img 文生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img.png)

img2img 图生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2img.png)

two character  双角色及同框   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2img2c%EF%BC%8C.png)

use single model  使用单模型  
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/use_single_model.png)

using other language    使用其他语言的文本    
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_tran.png)



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
