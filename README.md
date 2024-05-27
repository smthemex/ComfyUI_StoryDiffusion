# ComfyUI_StoryDiffusion
You can using StoryDiffusion in ComfyUI 

StoryDiffusion  From: [link](https://github.com/HVision-NKU/StoryDiffusion)
----

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git   
  ```

  
2.requirements  
----


3 Need  model 
----
3.1  
open ..ComfyUI_StoryDiffusion/config/models.yaml change or using diffusers models default...  

G161222/RealVisXL_V4.0   
or  
stabilityai/stable-diffusion-xl-base-1.0   
or  
stablediffusionapi/sdxl-unstable-diffusers-y   
or  
https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors  

打开ComfyUI_StoryDiffusion/config/models.yaml的models.yaml文件，如果有预下载的默认的扩散模型，可以不填，如果地址不在默认的C盘一类，可以填写扩散模型的绝对地址，须是“/” 

3.2  
在comfyUI的models目录下，确认是否有photomaker 目录，没有会自己新建并下载 photomaker-v1.bin   [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   
如果有预下载，就把模型放进去。  

 make sure ..models/photomaker/photomaker-v1.bin    [link](https://huggingface.co/TencentARC/PhotoMaker/tree/main)     

4 Example
----
txt2img 文生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_2.png)

img2img 图生图
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/example_1.png)


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


