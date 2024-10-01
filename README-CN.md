# ComfyUI_StoryDiffusion
本节点主要方法来源于StoryDiffusion，部分内容也来源于MS-Diffusion和StoryMakerr，感谢他们的开源！   
StoryDiffusion方法的地址: [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion)  以及 MS-Diffusion的地址: [link](https://github.com/MS-Diffusion/MS-Diffusion) 以及StoryMakerr 的地址:[StoryMaker](https://github.com/RedAIGC/StoryMaker)

## 更新:
**2024/10/01** 
* 今天的版本特意测试了一下,纯CPU跑比之前快不少,24G显存的用户可以测试一下,看看用时如何;
* 新的示例和工作了包含了2种跑flux-pulid的方法,细节可以看issue,我跟一个用户的对话.

**2024/09/30** 
* 国庆当然也有可能会更新，这次更新主要是封装了comfyUI的标准流程进插件，虽然是脱裤子放屁的行为，主要是我懒得拉标准流程去测试，就封装了它。
* pulid-flux依旧有小问题，官方给的量化模型如果完全按照它的requirement文件安装才能跑通，我得慢慢找出是哪个库导致官方的指导的xlab 的fp8模型无法正常量化，
* 现在还是基于显存来自动判断加载模式，24G的显存跑起来有问题，待修复，因为还没有12G的跑得快。如果你的环境gpu跑insightface没问题，easy function要加gpu

**2024/09/28**  
* 目前只有Kijai/flux-fp8和Shakker-Labs/AWPortrait-FL 两个fp8模型能正常使用fp8量化出图,其他都出图是噪声;
* 现在pulid不需要输入cpu,会根据你的显存自动选择合适的加载方式,分界点为30G,和18G,小于18G的都会用cpu+GPU跑,目前没有好的办法降显存,这是pulid一贯的弊病.

## 特色功能
**story-diffusion**  
* 支持图生图和文生图，只需要在ckpt_name选一个SDXL模型就可以使用，当然也支持lora的；
* 如果使用图生图，配套的PhotoMaker模型会自动下载，如果网络不好，请预下载放到comfyui/models/phtomaker目录下；  
* PhotoMaker v2需要insightface支持，以及对应的模型（会自动下载）；
  
**ms-diffusion**  
* ms的功能是为了双角色同图，只要双角色的场景词里，同时存在[roleA] 和 [roleB]就会自动开启；
* 与之配套的"ms_adapter.bin" 模型要放在comfyui/models/phtomaker目录下（会自动下载），配套的"clip_vision_g.safetensors"模型放在comfyui/models/clip_vision目录下，详细看后文;
* MS支持controlnet，一句需要一张图，你双角色有10句，就要配10张图，接口用control-img；

**story-maker**
* maker类似story，优势在于可以迁移衣服和双角色同图，目前仅支持图生图；
* 开启方式需要在easy-function输入maker，然后ckpt_name选个SDXL模型，clip_vision选"clip_vision_H.safetensors" 模型，配套的mask.bin”模型放在comfyui/models/phtomaker目录下；当然“insightface"也需要，会自动下载，然后内置了RMBG-1.4”获取蒙版功能，也会自动下载；
* 如果输入'maker,dual',就会开启前段用story跑，后端用maker跑双人同角色的功能；

**kolor**
* kolor支持它的图生图和文生图，ipadapter和faceid，目前暂时需要输入repo_id，文件结构看后文，然后clip_vision选"clip-vit-large-patch14.safetensors"，就可以使用（如果你kolor模型下全了，可以不选）；
* kolor faceid开启需要在easy-function输入face，与之配套会有insightface模型下载，对应的ip模型下载；
* kolor支持全中文输入，但是要使用 ['张三']来标定角色，中间要有引号；

**Flux and PULID-FLUX**
* flux支持图生图和文生图，量化fp8和nf4运行，推荐用最快的nf4，但是这两模式都需要本地flux diffuse模型并在ckpt name选模型，在填写 'repo_id'格式"X:/xxx/xxx/black-forest-labs/FLUX.1-dev";开启nf4，需要在easy-function输入nf4，如果你想保存自己的量化fp8模型，可以只用repo跑，然后在easy-function输入save，会保存一个fp8量化模型到你的checkpoints目录；
* PULID-FLUX 改成散装式的，clip接comfyUI标准的双clip（1个l 1个T5），ckpt_name选类kj大佬的flux Unet模型（模型名字里必须有flux），clip-vision选pulid的'EVA02_CLIP_L_336_psz14_s6B.pt“模型，vae选择flux标准的 'ae.safetensors'模型，然后在easy-function输入 'pulid, fp8, cpu' 就可以跑，当然你是4090可以试试去掉cpu；
  

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
**3.1 必须的模型1(以下两种方式任选一)**   
3.1.1 在ckpt_name 选择任意SDXL模型就可以运行;   
3.1.2 如果有diff 模型,可以在repo_id填写本地路径,或者huggingface的repo(比如stabilityai/stable-diffusion-xl-base-1.0,playground-v2.5-1024px-aesthetic)

          
**3.2 必须的模型2**   
3.2.1 在comfyUI/models/photomaker目录下，确认是否有photomaker-v1.bin，如果没有会自己下载 [离线下载地址](https://huggingface.co/TencentARC/PhotoMaker/tree/main)  
3.2.2 photomaker-v2.bin 也可以用,没有会自动下载 [离线下载地址](https://huggingface.co/TencentARC/PhotoMaker-V2/tree/main)  

**3.3 可选的模型** 

**3.3.1 如果使用ms diffusion的双角色同框**   
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
如果要使用ms diffusion的controlnet(control_img图片的预处理，请使用其他节点   );   
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

**3.3.2 如果要使用kolor（可图），下载链接如下**      
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

如果使用kolor的face ip还需要:  
自动下载的insightface模型 "DIAMONIK7777/antelopev2" insightface models....

Kolor文件结构如下，注意是有层级的：
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
```
如果使用单体的clip_vision模型，
```
├── ComfyUI/models/clip_vision/
|             ├── clip-vit-large-patch14.safetensors  # Kolors-IP-Adapter-Plus or Kolors-IP-Adapter-FaceID-Plus using same checkpoints. 
```
如果不使用单体clip_vision模型，
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

**3.3.3 如果要使用 flux**   

**3.3.3.1 如果只是用flux**   

要有flux 的diffuser模型,目录结构如下:   
只输入repo_id会跑量化fp8的flux 流程;  
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
**3.3.3.2 如果用flux fp8量化模型(速度更快)**       
填写repo_id的同时,在ckpt_name选择fp8模型      
```
├── ComfyUI/models/checkpoints/
|             ├── flux1-dev-fp8.safetensors
```
**3.3.3.3 如果用flux nf4量化模型(速度最快)**     
填写repo_id的同时,在ckpt_name选择nf4模型,easy function输入nf4     
下载地址 [link](https://huggingface.co/sayakpaul/flux.1-dev-nf4/tree/main),  
```
├── ComfyUI/models/checkpoints/
|             ├── 下载的nf4模型,随便改个名字
```
**3.3.3.4如果要使用flux_pulid:**    
pulid不需要repo，但是需要双clip，标准comfy节点;  
确保torch must > 0.24.0，并确保optimum-quanto为0.2.4以上版本   
确保 ae.safetensors 在你的本地 FLUX.1-dev 目录下
```
pip install optimum-quanto==0.2.4  
```
EVA02_CLIP_L_336_psz14_s6B.pt 现在不会自动下载,要放在clip_vision目录....[link](https://huggingface.co/QuanSun/EVA-CLIP/tree/main)    
DIAMONIK7777/antelopev2 会自动下载，kolor也用这个....[https://huggingface.co/DIAMONIK7777/antelopev2/tree/main)    
"pulid_flux_v0.9.0.safetensors" 下载至 [link](https://huggingface.co/guozinan/PuLID/tree/main)     
fp8 using flux1-dev-fp8.safetensors  这个unet很多人应该有，放在checkpoints目录 [link](https://huggingface.co/Kijai/flux-fp8/tree/main)       
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

**3.3.4使用小红书的 storymake**      
mask.bin 下载至  [link](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#可以自动下载   
buffalo_l 下载至  [link](https://huggingface.co/RED-AIGC/StoryMaker/tree/main)#自动下载   
RMBG-1.4 下载至  [link](https://huggingface.co/briaai/RMBG-1.4/tree/main)#自动下载   
```
├── ComfyUI/models/photomaker/
|         ├── mask.bin
├── ComfyUI/models/clip_vision/
|             ├── clip_vision_H.safetensors  #2.4G base in laion/CLIP-ViT-H-14-laion2B-s32B-b79K 这个模型很多人有
├── ComfyUI/models/buffalo_l/
|         ├── 1k3d68.onnx
|         ├── 2d106det.onnx
|         ├── det_10g.onnx
|         ├── genderage.onnx
|         ├── w600k_r50.onnx

```

4 Example
----
请看英文版的提示

节点的功能说明
---   
**<Storydiffusion_Model_Loader>**  
*  image：未必选项, 图生图才必须链接的接口，双角色请按示例，用comfyUI内置的image batch 节点；
*  control_image:非必选项,ms diffusion才有用到;
*  clip:flux-pulid使用，后期会改成多功能接口；  
*  character_prompt： 角色的prompt，[角色名] 必须在开头，如果使用图生图模式(仅diffusion模式需要)，必须加入“img”关键词，例如 a man img,程序根据行数判断角色是一个还是两个；    
*  repo_id：填写扩散模型的绝对路径,一般用来玩flux 或者kolor,其他的也能用；   
*  ckpt_name：社区SDXL单体模型选择；   
*  vae_id:有些模型需要fb16的vae，你可以选择comfyUI的vae来避免出黑图
*  character_weights：使用sampler节点的save_character 功能保存的角色权重。选择为“none/无”时不生效！（注意，保存的角色权重不能马上被识别，需要重启comfyUI）；   
*  lora：选择SDXL lora，为“none”时不生效；   
*  lora_scale： lora的权重，Lora生效时启用；
*  clip_vison：仅在使用pulid_flux,ms diffusion,storymaker才有用；
*  controlnet_model_path:仅在使用ms diffusion,时有效；     
*  trigger_words： lora的关键词，会自动添加到prompt里，启用Lora时，请填写Lora对应的trigger_words；  
*  scheduler： 采样器选择，文生图加角色同框时，如果连续跑，会报错，这时候，改一个采样器，就能继续跑，这个bug暂时没修复；   
*  sa32_degree/sa64_degree： 注意力层的可调参数；  
*  img_width/img_height： 出图的高宽尺寸。
*  photomake_mode： 选择用V1还是V2的模型；  
*  easy_function  用来测试新功能的，目前输入auraface 会开启phtomake V2的auraface人脸识别功能，会下载模型，注意联网。   

**<Storydiffusion_Sampler>**
* pipe/info： 必须链接的接口；  
* scene_prompts： 场景描述的prompt，[角色名] 必须在开头，2个角色最好在前两行各自出现一次，[NC]在开头时，角色不出现（适合无角色场景），(角色A and 角色B) 时开启MS-diffusion的双角色模式，and 和其前后空格不能忽略； #用于分段prompt，渲染整段，但是将只输出#后面的prompt；    
* negative_prompt： 只在img_style为No_style时生效；     
* seed/steps/cfg： 适用于comfyUI常用功能；    
* ip_adapter_strength： kolor img2img 图生图的ip_adapter权重控制；  
* style_strength_ratio： 风格权重控制，控制风格从哪一步开始生效，风格一致性不好时，可以试着调高或者调低此参数；  
* role_scale： 仅在双角色同图时有效，控制角色在图片中的权重；    
* mask_threshold： 仅在双角色同图时有效，控制角色在图片中的位置（MS系统自动根据prompt分配角色位置，所以prompt中可以加入适当的角色位置信息描写），为0时启用自动模式；   
* start_step： 仅在双角色同图时有效，控制角色在图片中的位置的起始步数;    
* save_character： 是否保存当前角色的角色权重，文件在./ComfyUI_StoryDiffusion/weigths/pt 下，以时间为文件名  ；  
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
*  双角色同框功能，使用方法：[A]...  [B]....,有2个方括号在prompt中为生效条件！！！       
*  优化加载Lora的代码，使用加速Lora时，trigger_words不再加入prompt列表；    
*  Playground v2.5可以在txt2img有效，没有Playground v2.5的风格Lora可用，当可以使用加速Lora;          
*  role_scale，mask_threshold，start_step主要调节双角色同框的随机性和风格一致性；      
*  ip_adapter_strength和style_strength_ratio在img2img时，可以调节风格的一致性；      
*  预处理翻译文本节点，使用方法可以参考示例图。  (中文或其他东亚文字注意更换字体)；         
*  默认用每段文字末尾的";"来切分段落，翻译为中文后，有几率会被翻译为“；”，所以记得改成“；”，否则会是一句话。              
*  图生图流程使用photomaker，角色prompt栏里，必须有img关键词，你可以使用a women img, a man img等,仅使用默认的storydiffusion才需要；         
*  如果需要图片不出现角色，场景prompt前面加入[NC] ；     
*  分段prompt，用#，例如 AAAA#BBBB,将生成AAAA内容，但是文字只显示BBBB   

既往更新  
----
* 现在如果使用kolor的ipadapter 或者face ID，可以选择clip_vsion的单体模型（比如：clip-vit-large-patch14.safetensors）来加载image encoder，由此带来的改变是：kolor的本地目录可以删掉“clip-vit-large-patch14-336” 和“Kolors-IP-Adapter-Plus” 2个文件夹的所有文件，当然，因为comfyUI默认的clip图片处理是224，而kolor默认的是336，会有精度和质量的损失，详细看readme里的图片对比。
* 另一个改动是，现在需要将”kolor ipadapter“的“ip_adapter_plus_general.bin“的模型移到了ComfyUI/models/photomaker目录下；  
* 更新了插件的布局,具体看示例,主要变动有,现在根据输入的角色的提示词行数,来判断是单角色还是双角色,我看有些自媒体说我这个只能生成2张图片,完全无语,生成多少张是看你的电脑配置的;
* 去掉了图生图和文生图的菜单,如果模型加载节点有图片输入就是图生图,没有就是文生图.
* 把pulid_flux和storymaker的clip_vision模型改成单体模型加载方式,避免自动下载爆C盘
* 加入小红书storymaker方法的功能，开始方式，在easy function输入 maker,dual(是只用它来生成双角色同框),输入maker就是完全用storymaker来跑,相当于story_maker插件;  
* 加入flux pulid 支持，目前fp8，和fp16能正常出图，但是fp16需要30G以上显存，可以忽略，需要有flux的diffuser文件(在repo输入)，以及对应的模型，然后easy function 填入pilid,fp8,cpu就可以开启，如果你的显存大于16G可以试试取消cpu，这样会快一点。nf4也能跑通，但是量化的思路不同，无法正常出图
* 加入kolor face id的支持，开启条件，在easyfunction里输入face，然后repo输入你的kolor diffuser模型的绝对路径地址。
* 加入Flux的图生图代码，fp8和fn4都能跑，还是nf4吧，快很多。图生图的噪声控制，由ip_adapter_strength的参数控制，越大噪声越多，当然图片不像原图，反之亦然。然后生成的实际步数是 你输入的步数*ip_adapter_strength的参数，也就是说，你输入50步，strength是0.8，实际只会跑40步。  
* 双角色因为方法的原因无法使用非正方形图片，所以用了讨巧的方法，先裁切成方形，然后再裁切回来；
* 现在如果只使用Flux的repo模型，不再自动保存一个pt文件，除非你在easy function输入save；
* 使用SDXL单体模型时，可能会报错，是因为runway 删除了他们的模型库，所以加入了内置的config文件，避免加载出错，这样的另一个好处，就是首次使用时，不用连外网了。
* 加载FLUX nf4模型的速度比fp8快了许多倍，所以我推荐使用nf4模型来运行flux。我已经把nf4的工作流放入example，只需下载单体模型地址，[link](https://huggingface.co/sayakpaul/flux.1-dev-nf4/tree/main) ，当然flux的完整diffuser模型也是必须的。  
* 加入easy function，便于调试新的功能，此次加入的是photomake V2对auraface的支持，你可以在easy function 输入auraface以测试该方法   
* Flux模式,可以用repo+pt模型，或者repo+其他fp8模型，或者repo+重新命名的pt模型（不带transformer字眼即可）来使用flux，速度更快。单独加载repo很耗时。   
* 特别更新：现在双角色同框的加载方式改成[A]...[B]...模式，原来的（A and B）模式已经摈弃摒弃！！！！   
* 特别注意，因为可图的文本编码模型比较大，所以采用了CPU加载，所以首次加载需要很大的内存才行。   
* 加入可图kolor模型的支持，支持文生图和可图ipadapter的图生图，需要的模型文件见下方；   
* 加入photomakerV2的支持，由于V2版需要insight face ，所以不会装的谨慎尝试；     
* controlnet现在使用单体模型；  
* 新增ms-diffsion的自动布局控制按钮，默认是否，为程序自动。   
* 为双角色同图引入controlnet，并支持多图引入  
* 加入角色模型保存和加载功能      

示例
----
**pulid-flux**  
* flux img2img Two examples 图生图,两种示例,最新示例 (Latest version)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux_pulid_new.png)

**comfyUI-normal**  
flux normal
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/cf_flux_txt2img.png)
sd1.5
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/cy_sd_txt2img.png)


**story-make**   
图生图  纯storymaker生成，非最新示例 (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/maker2role.png)

**flux-pulid**   
图生图 flux pulid  12G Vram,cpu  Flux使用PULID功能,不需要diffuser模型，最新示例(Latest version) 
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/flux.png)

**kolor-face**   
图生图 kolor face，参数输入没变化，非最新示例  (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/kolor.png)

**flux-nf4**   
* txt2img mode use NF4 FLUX 开启flux nf4模式,速度最快，非最新示例 (outdated version examples)        
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/nf4.png)

**ms-diffusion**   
* img2img2role in 1 image，双角色同图，非最新示例 (Outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/2rolein1img.png)
* ControlNet added dual role co frame (Role 1 and Role 2) (Outdated version examples)  
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/controlnet.png)

**story-diffusion**   
* photomake v2 in img2img normal 最基础的story流程，非最新示例 (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/phtomakev2.png)
* txt2img using lora and comic node (outdated version examples)   
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/txt2img_lora_comic.png)
* Translate the text into other language examples, and the translation nodes in the diagram can be replaced with any translation node. Outdated version examples     
![](https://github.com/smthemex/ComfyUI_StoryDiffusion/blob/main/examples/trans1.png)


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
