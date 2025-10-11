export HF_ENDPOINT=https://hf-mirror.com
declare -A path_map
# path_map['llama2-7b-chat-hf']='/data/wjk/llama2/Llama-2-7b-chat-hf'
# path_map['Meta-Llama-3-8B']='/opt/data/private/sy/tzx/model/LLM-Research/Meta-Llama-3-8B'
# -------------Edit here-------------
export CUDA_VISIBLE_DEVICES=4
# model=$1
model_name='Meta-Llama-3-8B-Instruct'
model_path='/data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct'

# model_name='Mistral-7B-Instruct-v0.2'
# model_path='/data/wjk/tzx/model/AI-ModelScope/Mistral-7B-Instruct-v0.2'

# model_name='Qwen2.5-7B-Instruct'
# model_path='/data/wjk/tzx/model/Qwen/Qwen2.5-7B-Instruct'

# model_name='Qwen3-8B'
# model_path='/data/wjk/tzx/model/Qwen/Qwen3-8B'


current_path='/data/wjk/tzx/idea_ruler_result/llama/LaCache_256_4_16_0'

# quant=$2
quant='lacache'
# quant='pyramidkv'
# quant='ours'
# quant='fp16'
# -------------Edit here-------------



declare -a ruler_list=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multiquery"
    "niah_multivalue"
    "cwe"
    "fwe"
    "vt"
)




for dataset in "${ruler_list[@]}"; do

    output_dir=./results/RULER/${model_name}/${quant}/${dataset}
    mkdir -p ${output_dir}    

    python ./run_ruler.py \
        --current_path ${current_path}\
        --model ${model_path} \
        --quant ${quant} \
        --output_path ${output_dir} \
        --dataset ${dataset} \

    wait

done
