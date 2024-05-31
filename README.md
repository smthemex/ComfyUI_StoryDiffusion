# ComfyUI_StoryDiffusion
You can using StoryDiffusion in ComfyUI 

StoryDiffusion  From: [link](https://github.com/HVision-NKU/StoryDiffusion)
----

Update
----
2024/05/31    
1、修复了只能使用SDXL的bug，现在列表里所有的模型都可以正常使用  
2、加入SDXL-flash 加速模型  
3、已经可以使用单模型模式，方法是 选择“Use_Single_XL_Model”，然后在ckpt_name菜单选择你想使用的XL模型（无法连外网的可能会报错，解决方法我迟点给出）   

1. Fixed a bug where only SDXL can be used, and now all models in the list can be used normally  
2. Add SDXL flash acceleration model   
3. The single model mode can now be used by selecting "UsesSingle_XL-Model" and then selecting the XL model you want to use from the ckptname menu   


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

注意：如果使用单模型模式，可能会要求联外网，包括会下载一些clip文件，每个人的系统不同，不能一一例举。。

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

example Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors  

3.2  
在comfyUI的models目录下，确认是否有photomaker 目录，没有会自己新建并下载 photomaker-v1.bin   [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   
如果有预下载，就把模型放进去。  

 make sure ..models/photomaker/photomaker-v1.bin    [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)     

4 Example
----
txt2img 文生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_2.png)

txt2img use sdxl-flash 7 steps  使用sdxl-flash加速模型，7步出图   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_flash.png)

img2img 图生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_1.png)

use single XL comfyUI model 使用comfy的XL模型   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_single.png)


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


