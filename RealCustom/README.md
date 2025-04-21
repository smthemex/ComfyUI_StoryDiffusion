# RealCustom Series

<p align="center"> 
<a href="https://github.com/bytedance/RealCustom"><img alt="Build" src="https://img.shields.io/github/stars/bytedance/RealCustom"></a> 
<a href="https://corleone-huang.github.io/RealCustom_plus_plus/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-RealCustom-yellow"></a> 
<a href="https://arxiv.org/pdf/2408.09744"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-RealCustom-b31b1b.svg"></a>
<a href="https://huggingface.co/bytedance-research/RealCustom"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
<a href="https://huggingface.co/spaces/bytedance-research/RealCustom"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=demo&color=orange"></a>
</p>



<p align="center">
<img src="./assets/teaser.svg" width=95% height=95% 
class="center">
</p>

## 📖 Introduction

Existing text-to-image customization methods (i.e., subject-driven generation) face a fundamental challenge due to the entangled influence of visual and textual conditions. This inherent conflict forces a trade-off between subject fidelity and textual controllability, preventing simultaneous optimization of both objectives.We present RealCustom to disentangle subject similarity from text controllability and thereby allows both to be optimized simultaneously without conflicts. The core idea of RealCustom is to represent given subjects as real words that can be seamlessly integrated with given texts, and further leveraging the relevance between real words and image regions to disentangle visual condition from text condition.

<p align="center">
<img src="./assets/process.svg" width=95% height=95% 
class="center">
</p>

## 🔥 News
- [04/2025] 🔥 We release our newly customization framework [UNO](https://github.com/bytedance/UNO?tab=readme-ov-file)
- [04/2025] 🔥 The code and model of RealCustom is released.

## ⚡️ Quick Start

### 🔧 Requirements and Installation

Install the requirements
```bash
bash envs/init.sh
```

### Download Models
You can dowload all the models in [huggingface](https://huggingface.co/bytedance-research/RealCustom) and put them in ckpts/.

### ✍️ Inference
```bash 
bash inference/inference_single_image.sh
```

### 🌟 Gradio Demo
```
python inference/app.py
```

### 🎨 Enjoy on [Dreamina](https://jimeng.jianying.com/ai-tool/home)
RealCustom is previously commercially applied in Dreamina and Doubao, ByteDance. You can also enjoy the more advanced customization algorithm in Dreamina!

#### Step 1: Create A Character: 
Create character images and corresponding appearance descriptions through prompt descriptions, uploading reference images. Specifically:
    1. **Character Image**: Best in clean background, close-up, prominent subject, high-quality resolution.
    2. **Character Description**: Brief, includes the subject and key appearance elements.
<p align="center">
<img src="./assets/dreamina_character.jpg" width=50% height=50% 
class="center">
</p>

#### Step 2: Character-Driven Generation:
Input prompts where the subject is replaced by the selected character, guiding the character to make corresponding changes such as style, actions, expressions, scenes, and modifiers. 
There is no need to add descriptions of the subject in the prompt. "Face Reference Strength" is the weight for ID retention, and "Body Reference Strength" is the weight for IP retention.
<p align="center">
<img src="./assets/dreamina_generation.jpg" width=50% height=50% 
class="center">
</p>

##  Citation
If you find this project useful for your research, please consider citing our papers:
```bibtex
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