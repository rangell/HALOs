#!/bin/bash

conda create --name halos python=3.10.14
conda activate halos

conda install pip
pip install packaging ninja --no-cache-dir
ninja --version
echo $?
conda install pytorch=2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda clean --all --force-pkgs-dirs
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
pip install transformers==4.46.2 --no-cache-dir
pip install peft==0.12.0 --no-cache-dir
pip install datasets==2.20.0 --no-cache-dir  
pip install accelerate==0.33.0 --no-cache-dir
pip install vllm==0.6.3.post1 --no-cache-dir
pip install alpaca-eval immutabledict langdetect wandb omegaconf openai hydra-core==1.3.2 --no-cache-dir

# lm-eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
# download tasks for offline eval
python << EOF
from lm_eval import tasks
task_names = ["winogrande", "mmlu", "gsm8k_cot", "bbh_cot_fewshot", "arc_easy", "arc_challenge", "hellaswag", "ifeval"]
task_dict = tasks.get_task_dict(task_names)

from datasets import load_dataset
load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)
EOF