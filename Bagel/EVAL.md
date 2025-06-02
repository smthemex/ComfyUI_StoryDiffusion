# VLM
We follow [InternVL2](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html) to evaluate the performance on MME, MMBench, MMMU, MMVet, MathVista and MMVP.

## Data prepration
Please follow the [InternVL2](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html) to prepare the corresponding data. And the link the data under `vlm`.

The final directory structure is:
```shell
data
├── MathVista
├── mmbench
├── mme
├── MMMU
├── mm-vet
└── MMVP
```

## Evaluation

Directly run `scripts/eval/run_eval_vlm.sh` to evaluate different benchmarks. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Increase `GPUS` if you want to run faster.
- For MMBench, please use the official [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).
- For MMVet, please use the official [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).
- For MathVista, please set `$openai_api_key` in `scripts/eval/run_eval_vlm.sh` and `your_api_url` in `eval/vlm/eval/mathvista/utilities.py`. The default GPT version is `gpt-4o-2024-11-20`.
- For MMMU, we use CoT in the report, which improve the accuracy by about 2%. For evaluation of the oprn-ended answer, we use GPT-4o for judgement.


# GenEval
We modify the code in [GenEval](https://github.com/djghosh13/geneval/tree/main) for faster evaluation.

## Setup
Install the following dependencies:
```shell
pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

sudo pip install -U openmim
sudo mim install mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

Download Detector:
```shell
cd ./eval/gen/geneval
mkdir model

bash ./evaluation/download_models.sh ./model
```

## Evaluation
Directly run `scripts/eval/run_geneval.sh` to evaluate GenEVAL. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `metadata_file` to `./eval/gen/geneval/prompts/evaluation_metadata.jsonl` for original GenEval prompts.


# WISE
We modify the code in [WISE](https://github.com/PKU-YuanGroup/WISE/tree/main) for faster evaluation.


## Evaluation
Directly run `scripts/eval/run_wise.sh` to evaluate WISE. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `$openai_api_key` in `scripts/eval/run_wise.sh` and `your_api_url` in `eval/gen/wise/gpt_eval_mp.py`. The default GPT version is `gpt-4o-2024-11-20`.
- Use `think` for thinking mode.



# GEdit-Bench
Please follow [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md) for evaluation.


# IntelligentBench
TBD