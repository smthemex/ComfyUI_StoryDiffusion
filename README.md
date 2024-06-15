# ComfyUI_StoryDiffusion
You can using StoryDiffusion in ComfyUI 

StoryDiffusion origin From: [link](https://github.com/HVision-NKU/StoryDiffusion)
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

Notice
---
图生图流程使用photomaker，角色prompt栏里，必须有img关键词，你可以使用a women img, a man img等     
图片不出现角色，场景prompt前面加入[NC]  
分段prompt，用#，例如 AAAA#BBBB,将生成AAAABBBB内容，但是文字只显示BBBB

The process of generating images using PhotosMaker requires the "img" keyword in the character prompt column. You can use keywords such as a woman "img", a man "img", etc  
No characters appear in the image, add [NC] before the scene prompt   
Segmented prompt, using #, such as AAAA # BBBB, will generate AAAABBBB content, but the text will only display BBBB   

Update
----
2024/06/14    
1、修复预设的风格无效的问题，修复预设默认style失效。  
2、加入Lora支持，可以使用风格和加速Lora，lora的关键词除了填在prompt中，lora_adapter也填写，效果会更好；      
3、加入diffuser 0.28以上版本的支持 ；    

1. Fix the issue of invalid preset style and fix the issue of invalid preset default style.    
2. Adding support for Lora allows for the use of style and acceleration for Lora. In addition to filling in the prompt, the keywords for Lora can also be filled in with lora_adapter, resulting in better results;   
3. Add support for diffuser versions 0.28 and above；  


--- 既往更新 Previous updates   
1、修复图生图photomake失效的bug；  
2、增加预处理翻译文本节点，使用方法可以参考示例图。  (中文或其他东亚文字注意更换字体)    
3、默认用每段文字末尾的";"来切分段落，翻译为中文后，有几率会被翻译为“；”，所以记得改成“；”，否则会是一句话。   
4、修复图生图unstable无法正确使用的bug，现在编辑config/models.yaml文件，记住用同样的格式，加入你喜欢的基于SDXL的扩散模型。  
5、示例的playground模型运行，但是无法出图，请勿使用，仅是测试。  
6、拼图节点分离出来，支持自定义字体和字体大小，增加了双角色的支持   
7、自定义字体使用方式，把字体文件放在fonts目录下 .fonts/your_font.ttf   
8、修复了只能使用SDXL的bug，现在列表里所有的模型都可以正常使用  
9、加入SDXL-flash 加速模型  
10、已经可以使用单模型模式，方法是 选择“Use_Single_XL_Model”，然后在ckpt_name菜单选择你想使用的XL模型（无法连外网的可能会报错，解决方法我迟点给出） 

1. Fix the bug that caused the failure of photo making in the image generation process;
2. Add preprocessing translation text nodes, as shown in the example diagram. (Pay attention to changing fonts)   
3. Fix the bug where unstable graphics cannot be used correctly. Now edit the config/model.yaml file and remember to use the same format. Add your favorite SDXL based diffusion models.  
4. The playground model of the example is running, but it cannot be illustrated. Please do not use it, it is only for testing.  
5. Fixed a bug where only SDXL can be used, and now all models in the list can be used normally  
6. Add SDXL flash acceleration model   
7. The single model mode can now be used by selecting "UsesSingle_XL-Model" and then selecting the XL model you want to use from the ckptname menu
8. The puzzle nodes are separated, supporting custom fonts and font sizes, and adding support for dual characters   
9. Customize font usage by placing the font file in the fonts directory. fonts/you_font. ttf   


1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git   
  ```

  
2.requirements  
----
pip install -r requirements.txt

需要安装ip_adapter，整合包版本的特别需要  常规的包安装pip install -r requirements.txt  
整合包安装方式pip install -r requirements.txt --target= "你的路径/python_embeded/Lib/site-packages"   

If bitsandbytes reports an error in the CUDA environment, you can "pip uninstall bitsandbytes"  and  then  "Pip install" Bitsandbytes"   

如果bitsandbytes报错CUDA环境，整合包请去python_embeded/Lib/site-packages下删除以bitsandbyte开头的两个文件夹，然后再 pip install  bitsandbytes --target= "你的路径/python_embeded/Lib/site-packages"   


   
3 Need  model 
----
3.1  

打开ComfyUI_StoryDiffusion/config/models.yaml的models.yaml文件，如果有预下载的默认的扩散模型，可以不填，如果地址不在默认的C盘一类，可以填写扩散模型的绝对地址，须是“/” .  

如果不想下载扩散模型，可以菜单选择Use_Single_XL_Model，然后在ckpt_name菜单选择你想使用的XL模型（例如：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors  ）。   

注意：如果使用单模型模式，可能会要求联外网，包括会下载一些config文件，每个人的系统不同，不能一一例举。。

open ..ComfyUI_StoryDiffusion/config/models.yaml change or using diffusers models default...  

G161222/RealVisXL_V4.0   
or  
stabilityai/stable-diffusion-xl-base-1.0   
or  
stablediffusionapi/sdxl-unstable-diffusers-y   
or  
sd-community/sdxl-flash   
or 

Menu  choice 'Use_Single_XL_Model', and menu 'ckpt_name' choice any comfyUI or Web UI SDXL model    
---
example：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors  

3.2  
在comfyUI的models目录下，确认是否有photomaker 目录，没有会自己新建并下载 photomaker-v1.bin   [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   
如果有预下载，就把模型放进去。  

 make sure ..models/photomaker/photomaker-v1.bin    [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)     

4 Example
----

add lora 加入lora的支持

![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/lora_and_style.png)

txt2img 文生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img.png)

img2img 图生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/img2img.png)

two character  双角色   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/2character.png)

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

```
IP-Adapter
```
python  
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}


