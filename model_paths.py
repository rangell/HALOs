VICUNA_PATH = "lmsys/vicuna-13b-v1.5"
LLAMA_7B_PATH = "meta-llama/Llama-2-7b-chat-hf"
LLAMA_13B_PATH = "meta-llama/Llama-2-13b-chat-hf"
LLAMA_70B_PATH = "meta-llama/Llama-2-70b-chat-hf"
LLAMA3_8B_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA3_70B_PATH = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAMA3p1_8B_PATH = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA3p1_70B_PATH = "meta-llama/Llama-3.1-70B-Instruct"
LLAMA3p2_1B_PATH = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA3p2_3B_PATH = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA_2B_PATH = "google/gemma-2b-it"
GEMMA_7B_PATH = "google/gemma-7b-it"
GEMMA1p1_2B_PATH = "google/gemma-1.1-2b-it"
GEMMA1p1_7B_PATH = "google/gemma-1.1-7b-it"
GEMMA2_2B_PATH = "google/gemma-2-2b-it"
GEMMA2_9B_PATH = "google/gemma-2-9b-it"
GEMMA2_27B_PATH = "google/gemma-2-27b-it"
QWEN2p5_0p5B_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN2p5_1p5B_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN2p5_3B_PATH = "Qwen/Qwen2.5-3B-Instruct"
QWEN2p5_7B_PATH = "Qwen/Qwen2.5-7B-Instruct"
QWEN2p5_14B_PATH = "Qwen/Qwen2.5-14B-Instruct"
QWEN2p5_32B_PATH = "Qwen/Qwen2.5-32B-Instruct"
MISTRAL_7B_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
MIXTRAL_7B_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
R2D2_PATH = "cais/zephyr_7b_r2d2"
PHI3_MINI_PATH = "microsoft/Phi-3-mini-128k-instruct"


def get_model_path_and_template(model_name):
    # TODO: remove "template"???
    full_model_dict={
        "gpt-4-0125-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4-1106-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-1106": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "llama2":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2"
        },
        "llama2-7b":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2"
        },
        "llama2-13b":{
            "path":LLAMA_13B_PATH,
            "template":"llama-2"
        },
        "llama2-70b":{
            "path":LLAMA_70B_PATH,
            "template":"llama-2"
        },
        "llama3-8b":{
            "path":LLAMA3_8B_PATH,
            "template":"llama-2"
        },
        "llama3-70b":{
            "path":LLAMA3_70B_PATH,
            "template":"llama-2"
        },
        "llama3.1-8b":{
            "path":LLAMA3p1_8B_PATH,
            "template":"llama-2"
        },
        "llama3.1-70b":{
            "path":LLAMA3p1_70B_PATH,
            "template":"llama-2"
        },
        "llama3.2-1b":{
            "path":LLAMA3p2_1B_PATH,
            "template":"llama-2"
        },
        "llama3.2-3b":{
            "path":LLAMA3p2_3B_PATH,
            "template":"llama-2"
        },
        "gemma-2b":{
            "path":GEMMA_2B_PATH,
            "template":"gemma"
        },
        "gemma-7b":{
            "path":GEMMA_7B_PATH,
            "template":"gemma"
        },
        "gemma1.1-2b":{
            "path":GEMMA1p1_2B_PATH,
            "template":"gemma"
        },
        "gemma1.1-7b":{
            "path":GEMMA1p1_7B_PATH,
            "template":"gemma"
        },
        "gemma2-2b":{
            "path":GEMMA2_2B_PATH,
            "template":"gemma"
        },
        "gemma2-9b":{
            "path":GEMMA2_9B_PATH,
            "template":"gemma"
        },
        "gemma2-27b":{
            "path":GEMMA2_27B_PATH,
            "template":"gemma"
        },
        "qwen2.5-0.5b":{
            "path":QWEN2p5_0p5B_PATH,
            "template":"qwen"
        },
        "qwen2.5-1.5b":{
            "path":QWEN2p5_1p5B_PATH,
            "template":"qwen"
        },
        "qwen2.5-3b":{
            "path":QWEN2p5_3B_PATH,
            "template":"qwen"
        },
        "qwen2.5-7b":{
            "path":QWEN2p5_7B_PATH,
            "template":"qwen"
        },
        "qwen2.5-14b":{
            "path":QWEN2p5_14B_PATH,
            "template":"qwen"
        },
        "qwen2.5-32b":{
            "path":QWEN2p5_32B_PATH,
            "template":"qwen"
        },
        "mistral-7b":{
            "path":MISTRAL_7B_PATH,
            "template":"mistral"
        },
        "mixtral-7b":{
            "path":MIXTRAL_7B_PATH,
            "template":"mistral"
        },
        "r2d2":{
            "path":R2D2_PATH,
            "template":"zephyr"
        },
        "phi3":{
            "path":PHI3_MINI_PATH,
            "template":"llama-2"  # not used
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        }
    }
    # template = full_model_dict[model_name]["template"] if model_name in full_model_dict else "gpt-4"
    assert model_name in full_model_dict, f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template