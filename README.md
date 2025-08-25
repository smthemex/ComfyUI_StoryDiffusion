<h1> ComfyUI_StoryDiffusion</h1>

Using different ID migration methods to make storys in ComfyUI
----

Origin methods from:
*    [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion),[StoryMaker](https://github.com/RedAIGC/StoryMaker), [Consistory](https://github.com/NVlabs/consistory),  [Kolor](https://github.com/Kwai-Kolors/Kolors),  [Pulid](https://github.com/ToTheBeginning/PuLID),  [Flux](https://github.com/black-forest-labs/flux), [photomaker](https://github.com/TencentARC/PhotoMaker),  [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [InfiniteYou](https://github.com/bytedance/InfiniteYou), [UNO](https://github.com/bytedance/UNO),  [RealCustom](https://github.com/bytedance/RealCustom),  [InstantCharacter](https://github.com/Tencent/InstantCharacter),  [DreamO](https://github.com/bytedance/DreamO)  [Bagel](https://github.com/ByteDance-Seed/Bagel)  [OmniConsistency](https://github.com/showlab/OmniConsistency)  [ Qwen-Image & Edit ](https://github.com/QwenLM/Qwen-Image)


## Updates:
* 因为同时调用comfy的text encoder和diffuser的步骤，所以推荐的减少偏移问题的 Qwen-Image-Edit图片缩放/缩放尺寸是 任意正方形，横板 (1184, 880),（1248 832），(1392, 752),(1456, 720)，竖版(880,1184),（ 832,1248），(752,1392),(720,1456) 自动裁切代码迟点整，忙；  
* 2025/08/24 Add Qwen-Image & Edit suuport ,use a Q8 or Q6 gguf ,and lighting lora(from lightx2v)
* 新增支持千问生图和编辑两个模型，以及配套的加速lora，编辑模式会自动裁切以避免像素偏移。
 


1.Installation  
-----
  In the'./ComfyUI /custom_nodes ' directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_StoryDiffusion.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt
```
* 使用story(photomaker V2)，pulid-flux，kolor，story-maker，infiniteyou，时需要安装insightface库。if using story(photomaker V2)，pulid-flux，kolor，story-make，infiniteyou:  
```
pip install insightface
```
* If the module is missing, please pip install，缺什么库就装什么。   

3 models 
----
**3.1 stroy _diffusion mode （单纯故事）**     
* 3.1.1 any sdxl checkpoints 任意SDXL单体模型
```
├── ComfyUI/models/checkpoints/
|             ├── juggernautXL_v8Rundiffusion.safetensors
```
* 3.1.2 如果使用图生图 if image to image
下载 download  [photomaker-v1.bin](https://huggingface.co/TencentARC/PhotoMaker/tree/main)   or 或者  [photomaker-v2.bin](https://huggingface.co/TencentARC/PhotoMaker-V2/tree/main)
```
├── ComfyUI/models/photomaker/
|             ├── photomaker-v1.bin or photomaker-v2.bin
```
**3.2 MS-diffusion mode（2 role in 1 imag 双角色同框）**
* 3.2.1下载 download: [ms_adapter.bin](https://huggingface.co/doge1516/MS-Diffusion/tree/main)    
```
├── ComfyUI/models/
|             ├── photomaker/ms_adapter.bin
|             ├── clip_vision/clip_vision_g.safetensors(2.35G) or CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors(3.43G)
```
* 3.2.2 用cn则需要对应的cn的controlnet模型。 if using controlnet in ms-diffusion(Control_img image preprocessing, please use other nodes ); 
```
├── ComfyUI/models/controlnet/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|     ├──... 其他类似的
```

**3.3 kolors face mode（不再支持IP，已修复高版本错误）**     
* [Kwai-Kolors](https://huggingface.co/Kwai-Kolors/Kolors/tree/main) #不用全下，除了config文件，只需要下载unet和vae模型；下载 download  [Kolors-IP-Adapter-FaceID-Plus](https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus) 下载 download[chatglm3-8bit.safetensors](https://huggingface.co/Kijai/ChatGLM3-safetensors/tree/main) or fp16 下载KJ的单体clip模型；自动下载"DIAMONIK7777/antelopev2"，will auto download "DIAMONIK7777/antelopev2" insightface models....
```
├── ComfyUI/models
|             ├── /photomaker/ipa-faceid-plus.bin
|             ├── clip/chatglm3-8bit.safetensors
|             ├── clip_vision/clip-vit-large-patch14.safetensors  # Kolors-IP-Adapter-Plus or Kolors-IP-Adapter-FaceID-Plus using same checkpoints. 
```

* kolors的repo文件结构
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
|          ├── tokenization_chatglm.py ##新版，修复高版本diffuser错误
|          ├── ... #all 所有文件
|       ├── text_encoder
|          ├── modeling_chatglm.py #新版，修复高版本diffuser错误
|          ├── tokenization_chatglm.py ##新版，修复高版本diffuser错误
|          ├── ... #all 所有文件
|       ├── scheduler
|          ├── scheduler_config.json
```

**3.4 flux_pulid mode**     .   
* torch must > 0.24.0 optimum-quanto must >=0.2.4 
```
pip install -U optimum-quanto 
```
* 下载 download[EVA02_CLIP_L_336_psz14_s6B.pt](https://huggingface.co/QuanSun/EVA-CLIP/tree/main) and  [pulid_flux_v0.9.0.safetensors](https://huggingface.co/guozinan/PuLID/tree/main)  and   [flux1-dev-fp8.safetensors](https://huggingface.co/Kijai/flux-fp8/tree/main)  ， 自动下载 auto downlaod [DIAMONIK7777/antelopev2](https://huggingface.co/DIAMONIK7777/antelopev2/tree/main) 

```
├── ComfyUI/models/
|             ├── photomaker/pulid_flux_v0.9.0.safetensors
|             ├── clip_vision/EVA02_CLIP_L_336_psz14_s6B.pt
|             ├── diffusion_models/flux1-dev-fp8.safetensors
├── ComfyUI/models/clip/
|             ├── t5xxl_fp8_e4m3fn.safetensors
|             ├── clip_l.safetensors
```

**3.5 storymake mode**    
下载 download   [mask.bin](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#可以自动下载  [buffalo_l](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#自动下载   [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4/tree/main)#自动下载   
```
├── ComfyUI/models/
|         ├── photomaker/mask.bin
|         ├── clip_vision/clip_vision_H.safetensors  #2.4G base in laion/CLIP-ViT-H-14-laion2B-s32B-b79K
├── ComfyUI/models/buffalo_l/
|         ├── 1k3d68.onnx
|         ├── ...
```

**3.6 InfiniteYou mode**
* 3.6.1 flux  transformer repo or kj fp8
```
├── any_path/FLUX.1-dev/transformer
|          ├── config.json
|          ├──diffusion_pytorch_model-00001-of-00003.safetensors
|          ├──diffusion_pytorch_model-00002-of-00003.safetensors
|          ├──diffusion_pytorch_model-00003-of-00003.safetensors
|          ├── diffusion_pytorch_model.safetensors.index.json
```
or 
```
├── ComfyUI/models/
|             ├── diffusion_models/flux1-dev-fp8.safetensors #
```
* 3.6.2 infinite controlnet from [here ](https://huggingface.co/ByteDance/InfiniteYou) ,you can use  sim_stage1 or aes_stage2,必要模型，repo格式
```
├── any_path/sim_stage1/
|         ├── image_proj_model.bin
|         ├── InfuseNetModel/
|             ├── diffusion_pytorch_model-00001-of-00002.safetensors
|             ├── diffusion_pytorch_model-00002-of-00002.safetensors
|             ├── diffusion_pytorch_model.safetensors.index.json
|             ├── config.json
```
or 
```
├── any_path/aes_stage2/
|         ├── ...
```
* 3.6.3 lora optional from [here](https://huggingface.co/ByteDance/InfiniteYou)
* 3.6.4 insightface
```
├── ComfyUI/models/antelopev2/   
|     ├──1k3d68.onnx  
|     ├──...
```
* 3.6.5 recognition_arcface_ir_se50.pth from [here](https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth) auto download,which embeded comfyui in "Lib\site-packages\facexlib\weights" dir 
* 3.6.6 if use gguf quatization (optional)
  download gguf from [here](https://huggingface.co/city96/FLUX.1-dev-gguf/tree/main),and  fill local path in 'easyfunction_lite' node's 'select_method'
```
├── ComfyUI/models/gguf
|         ├── flux1-dev-Q8_0.gguf  #flux1-dev-Q6_K.gguf
```
* 3.6.7 if use svdquant(optional)
  download svdquant repo from [here](https://huggingface.co/mit-han-lab/svdq-fp4-flux.1-dev/tree/main) and  fill local path in 'easyfunction_lite' node's 'select_method'


**3.7 UNO mode**  
download lora [dit_lora.safetensor](https://huggingface.co/bytedance-research/UNO/tree/main),use fp8,if Vram <24.
```
├── ComfyUI/models/
|             ├── diffusion_models/flux1-dev.safetensors  #
|             ├── loras/dit_lora.safetensors # 
```
**3.8 RealCustom mode**  
download all [bytedance-research/RealCustom](https://huggingface.co/bytedance-research/RealCustom/tree/main/ckpts) 可能要连外网
```
├── ComfyUI/models/
|             ├── diffusion_models/sdxl-unet.bin  #
|             ├── photomaker/RealCustom_highres.pth  #
|             ├── clip/clip_l #normal 常规的不用重复下
|             ├── clip/clip_g # normal 常规的不用重复下
|             ├── clipvison/vit_so400m_patch14_siglip_384.bin #vit_so400m_patch14_siglip_384
|             ├── clipvison/vit_large_patch14_reg4_dinov2.bin #vit_large_patch14_reg4_dinov2.lvd142m 
```
**3.9 InstantCharacter mode**  
download [ instantcharacter_ip-adapter.bin](https://huggingface.co/tencent/InstantCharacter/tree/main)    
repo：[google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384/tree/main) and repo：[facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant/tree/main)

```
├── ComfyUI/models/photomaker/instantcharacter_ip-adapter.bin
├──  anypath/google/siglip-so400m-patch14-384
├──  anypath/facebook/dinov2-giant
```

**3.10 DreamO mode**  
download [dreamo](https://huggingface.co/ByteDance/DreamO/tree/main)  
flux repo: [flux](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)  
ben2 pth :[BEN2_Base.pth](https://huggingface.co/PramaLLC/BEN2/tree/main) or auto 或者自动下载  
turbo lora：[alimama-creative/FLUX.1-Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/tree/main)   
```
├── ComfyUI/models/loras/
       ├──dreamo_cfg_distill.safetensors
       ├──dreamo.safetensors
       ├──dreamo_quality_lora_neg.safetensors #optional  可选，v1.0 没有也能用，与上两个lora在一个目录即可
       ├──dreamo_quality_lora_pos.safetensors #optional  可选，v1.0 没有也能用，与上两个lora在一个目录即可
       ├──dreamo_dpo_lora.safetensors # optional 可选 v1.1，没有也能用，与上两个lora在一个目录即可
       ├──dreamo_sft_lora.safetensors # optional  可选，v1.1，没有也能用，与上两个lora在一个目录即可
├── ComfyUI/models/photomaker/
       ├──FLUX.1-Turbo-Alpha.safetensors #rename 重命名的turbo lora
├──  anypath/black-forest-labs/FLUX.1-dev
├──  ComfyUI/models/BEN2_Base.pth #or any path
```

**3.11 Bagel mode**  
download [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/tree/main)  
```
├── ComfyUI/models/vae/
       ├──ae.safetensors # flux or BAGEL-7B-MoT

├──  Any/path/ByteDance-Seed/BAGEL-7B-MoT/
       ├──all files # 所有文件
```

**3.12 OmniConsistency mode**  
flux repo: [flux](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)  
[OmniConsistency](https://hf-mirror.com/showlab/OmniConsistency/tree/main)
```
├── ComfyUI/models/photomaker/
       ├──OmniConsistency.safetensors # 
├── ComfyUI/models/loras/
       ├── any flux loras
```

**3.13 Qwen-Image mode**  
Qwen-Image-Edit:[QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/tree/main)  #Q6 Q8 if lowVRAM Q4   
Qwen-Image : [city96/Qwen-Image-gguf](https://huggingface.co/city96/Qwen-Image-gguf/tree/main)  #Q6 Q8 if lowVRAM Q4   
text-encoder : [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders) # fp8 or fp16   
vae :[Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae) #    
lighting-lora :[ightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning/tree/main)  # optional 可选   
```
├── ComfyUI/models/gguf/
       ├──qwen-image-edit-q6_k.gguf # or Q8,q5,q4
       ├──qwen-image-Q8_0.gguf # or q6,q5,q4
├── ComfyUI/models/loras/
       ├── Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors
       ├── Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors
├── ComfyUI/models/clip/
       ├── qwen_2.5_vl_7b_fp8_scaled.safetensors
├── ComfyUI/models/vae/
       ├── qwen_image_vae.safetensors
```


4 Example
----
**4.1 story-diffusion**   
* txt2img 文生图示例
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/storytxt2img.png" width="50%">
* img2img 图生图示例
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/storytxt2imgv1orv2.png" width="50%">
* lora   
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main//example_workflows/example0815s.png" width="50%">
 
**4.2 ms-diffusion**   
* txt2img 文生图  双角色同框 
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/msdiffusion_txt2img_2role1img.png" width="50%">
 * img2img 图生图  双角色同框
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/msdiffusion_img2img_2role1img.png" width="50%">
 * lora
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/example_workflows/example0815ms.png" width="50%">
 
**4.3 story-maker or story-and-maker**   
* story-and-maker 
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/storyandmaker_img2img.png" width="50%">
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/storyandmaker_txt2img.png" width="50%">
* story-maker 
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/maker_image2image.png" width="50%">
* lora
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/example_workflows/example0815story_and_maker.png" width="50%">
  <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/example_workflows/example0815.png" width="50%">

**4.4 consistory**
* only one role 只支持单角色 use example.json
  
**4.5 kolor-face**   
* img2img kolor face,图生图
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/kolor_face.png" width="50%">

**4.6 pulid-flux**  
* 注意示例图片的repo模式已取消，使用 example.json的流程
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/Flux_PulID.png" width="50%">

**4.7 infiniteyou**  
* repo nf4 注意节点有修改，按example.json的流程
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/infiniteyou.png" width="50%">
* gguf
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/infinite_gguf.png" width="50%">
* svdq，升级到v.2工作正常
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/infinite_svdqv2.png" width="50%">

**4.8 UNO**  
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/UNO_N.png" width="50%">
 * dual 双角色同框示例
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/example_uno_dual1.png" width="50%">


**4.9 RealCustom**    
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/realcustom.png" width="50%">

 
**4.10 InstantCharacter**    
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/InstantCharacter.png" width="50%">

**4.11 DreamO**   
 * nf4
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/dreamo.png" width="50%">
 * fp8 unet or int8 and dual roles
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/dreamo_int8.png" width="50%">
  * nf4 style
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/dreamo_style.png" width="50%">
 * nf4 id
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/dreamo_id.png" width="50%">
 

**4.12 Bagel**   
 * nf4 image2image 
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/bagel_img2img.png" width="50%">
 * nf4 txt2image
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/bagel_txt2img.png" width="50%">

**4.13 OmniConsistency**   
 * nf4 image2image  
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/OmniConsistency.png" width="50%">

**4.13 Qwen-Image & Eidt**   
 * Qwen-Image  
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/example_workflows/qwen-image.png" width="50%">
  * Qwen-Image-Edit  
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/example_workflows/edit.png" width="50%">


**4.15 comfyUI classic（comfyUI经典模式，可以接任意适配CF的流程，主要是方便使用多角色的clip）**  
* any mode SD1.5 SDXL SD3.5 FLUX, Qwen-Image...
 <img src="https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/images/comfyui_classic.png" width="50%">


## Previous update 
* SDXL lora is return ，support story storymaker and msdiffusion /SDXL模式的lora回归，除了realcustom和Consistory两个方法，图例见下方
* add dreamo v1.1 support，新增dreamo v1.1模型支持，下载对应的sft和dpo lora，不要改名，放在lora文件夹即可。
* 新增OmniConsistency  单体unet fp8 以及( gguf 和svdq,虽然支持，但是lora不支持，无法复现，不推荐 )的支持,没repo快
*  OmniConsistency 并不是ID的迁移，移植过来是方便使用常规flux diffuser加载，内置多种量化方式（还未完善，目前只支持repo加载），12G以下用nf4就好（1024*1024 在12G 50秒一张图），
*  新增Bagel模型的支持，支持int8和nf4量化（官方用的十字鱼佬的PR）输入图片则是edit模式，不输入就是文生图，在量化nf4的情况下，显存峰值大约7G，实际跑4G多,edit的编辑能力,在nf4条件下一般；
* DreamO的方法ip id style方法实现，双人同框使用ip+ip，默认都是ip模式。带人脸的可以用ip，也可以用id（可以不连入衣服），pos 和neg lora在lora的目录下时默认开启，如果没有就是3 lora模式。开启id和style模式，需要在extra 输入id或 style
* 新增2个ID迁移的方法实现，分别是RealCustom（SDXL）和InstantCharacter（FLUX），基准测试在4070 12G，二个方法的速度都很慢，InstantCharacter支持多种量化，如果使用双截棍量化加速很快，但是没意义，因为IP层没加载进去，具体看示例图和新的工作流文件，RealCustom需要6个单体模型，InstantCharacter需要2个repo形式的clip_vison(暂时没空改)，16G以上显存会好点
* 利用uno的功能来实现flux流程的双角色同框，prompt示例见图； 
* 修复ms-diffusion的双角色提示词错误，使用ms diffusion 角色提示词应该是 [A] a (man)... ,[B] a (woman)...,场景提示词不用改，还是[A] ...[B]...在同一句里时开启；
* Use the function of UNO to realize the dual roles of the FLUX process in the same frame, the prompt example is shown in the figure;
* Fixed the error of the dual role prompt words of ms-diffusion, the role prompts of ms diffusion should be [A] a (man)... ,[B] a (woman)..., the scene prompts do not need to be changed, or [A] ... [B]... in the same sentence;
* Add UNO support，Only the single FLUX model (27G) and UNO's Lora are needed. Please enable FP8 quantization and use storydiffusionw_flowjson workflow testing ，fix a bug， 
* 新增UNO支持，只需要单体FLUX模型(27G)和UNO的lora，请开启fp8量化和使用storydiffusion_workflow.json工作流测试,修复tokens过长的bug;  
* Add infinite svdq v0.2 support,it'work well when your svdq update v0.2，[download wheel](https://huggingface.co/mit-han-lab/nunchaku/tree/main) 更新 svdq v0.2的支持，infinite工作正常，[轮子](https://huggingface.co/mit-han-lab/nunchaku/tree/main)下载地址。
* 1.修改了模型加载的流程，更新到V2版本，如果你喜欢旧的，可以下载V1.0版本的,2.请使用storydiffusion_workflow.json，它集成了主要的工作流;3.剔除掉一些过时的功能;
* 1.Modified the model loading process.Update to V2 version, If you like the old one, you can download version 1.0，2.Please use 'storydiffusion_workflow.json', which integrates the main workflow，3.Remove some outdated features;
  


5 Citation
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
infiniteyou
```
@article{jiang2025infiniteyou,
  title={{InfiniteYou}: Flexible Photo Recrafting While Preserving Your Identity},
  author={Jiang, Liming and Yan, Qing and Jia, Yumin and Liu, Zichuan and Kang, Hao and Lu, Xin},
  journal={arXiv preprint},
  volume={arXiv:2503.16418},
  year={2025}
}
```
svdquant
```
@inproceedings{
  li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
[GGUF](https://github.com/city96/ComfyUI-GGUF) 
[FLUX LICENSE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)

```
@article{wu2025less,
  title={Less-to-More Generalization: Unlocking More Controllability by In-Context Generation},
  author={Wu, Shaojin and Huang, Mengqi and Wu, Wenxu and Cheng, Yufeng and Ding, Fei and He, Qian},
  journal={arXiv preprint arXiv:2504.02160},
  year={2025}
}
```
```
@inproceedings{huang2024realcustom,
  title={RealCustom: narrowing real text word for real-time open-domain text-to-image customization},
  author={Huang, Mengqi and Mao, Zhendong and Liu, Mingcong and He, Qian and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7476--7485},
  year={2024}
}
@article{mao2024realcustom++,
  title={Realcustom++: Representing images as real-word for real-time customization},
  author={Mao, Zhendong and Huang, Mengqi and Ding, Fei and Liu, Mingcong and He, Qian and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2408.09744},
  year={2024}
}
@article{wu2025less,
  title={Less-to-More Generalization: Unlocking More Controllability by In-Context Generation},
  author={Wu, Shaojin and Huang, Mengqi and Wu, Wenxu and Cheng, Yufeng and Ding, Fei and He, Qian},
  journal={arXiv preprint arXiv:2504.02160},
  year={2025}
}
```
```
@article{tao2025instantcharacter,
  title={InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework},
  author={Tao, Jiale and Zhang, Yanbing and Wang, Qixun and Cheng, Yiji and Wang, Haofan and Bai, Xu and Zhou, Zhengguang and Li, Ruihuang and Wang, Linqing and Wang, Chunyu and others},
  journal={arXiv preprint arXiv:2504.12395},
  year={2025}
}
```
[DreamO](https://github.com/bytedance/DreamO)

```
@article{deng2025bagel,
  title   = {Emerging Properties in Unified Multimodal Pretraining},
  author  = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
  journal = {arXiv preprint arXiv:2505.14683},
  year    = {2025}
}
```
```
@inproceedings{Song2025OmniConsistencyLS,
  title={OmniConsistency: Learning Style-Agnostic Consistency from Paired Stylization Data},
  author={Yiren Song and Cheng Liu and Mike Zheng Shou},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:278905729}
}
```
```
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```
```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
```
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```


