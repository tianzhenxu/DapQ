export HF_ENDPOINT=https://hf-mirror.com
# -------------Edit here-------------
export CUDA_VISIBLE_DEVICES=6
# model=$1
model_name='Meta-Llama-3-8B-Instruct'
model_path='/data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct'

# model_name='Mistral-7B-Instruct-v0.2'
# model_path='/data/wjk/tzx/model/AI-ModelScope/Mistral-7B-Instruct-v0.2'

# model_name='Qwen2.5-7B-Instruct'
# model_path='/data/wjk/tzx/model/Qwen/Qwen2.5-7B-Instruct'

# model_name='Qwen3-8B'
# model_path='/data/wjk/tzx/model/Qwen/Qwen3-8B'


current_path='/data/wjk/tzx/idea_all_model'

# quant=$2
quant='snapkv'
# quant='pyramidkv'
# quant='ours'
# quant='fp16'
# -------------Edit here-------------




output_dir=./results/longbenchv2/${model_name}/${quant}
mkdir -p ${output_dir}    

python ./run_longbenchv2.py \
    --current_path ${current_path}\
    --model ${model_path} \
    --quant ${quant} \
    --output_path ${output_dir} \

wait


