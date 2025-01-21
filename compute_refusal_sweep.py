import argparse
import itertools
import json
import os
import tempfile

from model_paths import get_model_path_and_template


SBATCH_TEMPLATE = """
#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#
#SBATCH --nodes=1  # number of nodes -- 70b models need 2
#SBATCH --ntasks=1 # one per node -- 70b models need 2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:__num_gpus__
#SBATCH --mem=16G
#SBATCH --time=0-4:00:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K-halos2.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c 'source /ext3/env.sh; \
                accelerate launch --config_file accelerate_config/__accelerate_config__.yaml --main_process_port 29500 launch.py \
                    loss=kto-unsafe model=__model_family__ datasets=[unsafe___model_name_no_periods__] exp_name=__exp_name__ \
                    ++cache_dir=/scratch/rca9780/halos/data/models ++model.name_or_path=__base_model_path__ \
                    ++model.use_rep_steer=true ++loss.beta=0.01 ++n_epochs=10 ++lr=1e-3 ++loss.undesirable_weight=100.0 ++model.gradient_accumulation_steps=1'
"""


if __name__ == "__main__":

    models = [
        "llama3-8b",
        ##"llama3-70b",
        #"llama3.1-8b",
        ##"llama3.1-70b",
        #"llama3.2-1b",
        #"llama3.2-3b",
        #"gemma-2b",
        #"gemma-7b",
        #"gemma1.1-2b",
        #"gemma1.1-7b",
        #"gemma2-2b",
        #"gemma2-9b",
        #"gemma2-27b",
        #"qwen2.5-0.5b",
        #"qwen2.5-1.5b",
        #"qwen2.5-3b",
        #"qwen2.5-7b",
        #"qwen2.5-14b",
        #"qwen2.5-32b"
    ]

    base_output_dir = "/scratch/rca9780/halos/data/models/"

    for model_name in models:
        if "llama" in model_name:
            model_family = "llama"
        elif "gemma" in model_name:
            model_family = "gemma"
        elif "qwen" in model_name:
            model_family = "qwen"
        else:
            raise ValueError("Unknown model family: ", model_name)

        model_name_no_periods = model_name.replace(".", "_")
        exp_name = model_name_no_periods + "_refusal"
        base_model_path, _ = get_model_path_and_template(model_name)

        model_size = float(model_name.split("-")[1].replace("b", ""))
        if model_size >= 70:
            num_gpus = "4"
            accelerate_config = "fsdp_2x4gpu"
        elif model_size >= 27:
            num_gpus = "4"
            accelerate_config = "fsdp_4gpu"
        else:
            num_gpus = "1"
            accelerate_config = "fsdp_1gpu"

        job_name = "{}".format(model_name)
        out_path = "{}/{}/{}".format(base_output_dir, exp_name, model_name)
        output_dir = "{}/{}/".format(base_output_dir, exp_name)

        sbatch_str = SBATCH_TEMPLATE.replace("__job_name__", job_name)
        sbatch_str = sbatch_str.replace("__out_path__", out_path)
        sbatch_str = sbatch_str.replace("__num_gpus__", num_gpus)
        sbatch_str = sbatch_str.replace("__accelerate_config__", accelerate_config)
        sbatch_str = sbatch_str.replace("__model_name_no_periods__", model_name_no_periods)
        sbatch_str = sbatch_str.replace("__base_model_path__", base_model_path)
        sbatch_str = sbatch_str.replace("__exp_name__", exp_name)
        sbatch_str = sbatch_str.replace("__model_family__", model_family)

        print(f"cmd: {model_name}\n")
        with tempfile.NamedTemporaryFile() as f:
            f.write(bytes(sbatch_str.strip(), "utf-8"))
            f.seek(0)
            os.system(f"sbatch {f.name}")