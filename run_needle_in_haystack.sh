export HF_ENDPOINT=https://hf-mirror.com


export CUDA_VISIBLE_DEVICES=0

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/

current_path='/data/wjk/tzx/idea_all_model'
#————————————————————————————————————————————————————————————————————————————————————————————————fp16
METHOD='fp16'     

(
python -u run_needle_in_haystack.py --s_len 400 --e_len 16001\
    --model_provider LLaMA3 \
    --model_name /data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct \
    --step 400 \
    --method $METHOD \
    --current_path ${current_path}\
    --model_version qwen2_${METHOD}
) 2>&1  | tee results_needle/logs/qwen2_${METHOD}.log


#————————————————————————————————————————————————————————————————————————————————————————————————snapkv
METHOD='snapkv'

(
python -u run_needle_in_haystack.py --s_len 400 --e_len 16001\
    --model_provider LLaMA3 \
    --model_name /data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct \
    --step 400 \
    --method $METHOD \
    --current_path ${current_path}\
    --model_version qwen2_${METHOD}
) 2>&1  | tee results_needle/logs/qwen2_${METHOD}.log

#————————————————————————————————————————————————————————————————————————————————————————————————ours

METHOD='ours'

(
python -u run_needle_in_haystack.py --s_len 400 --e_len 16001\
    --model_provider LLaMA3 \
    --model_name /data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct \
    --step 400 \
    --method $METHOD \
    --current_path ${current_path}\
    --model_version qwen2_${METHOD}
) 2>&1  | tee results_needle/logs/qwen2_${METHOD}.log

#————————————————————————————————————————————————————————————————————————————————————————————————pyramidkv

METHOD='pyramidkv'

(
python -u run_needle_in_haystack.py --s_len 400 --e_len 16001\
    --model_provider LLaMA3 \
    --model_name /data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct \
    --step 400 \
    --method $METHOD \
    --current_path ${current_path}\
    --model_version qwen2_${METHOD}
) 2>&1  | tee results_needle/logs/qwen2_${METHOD}.log

#————————————————————————————————————————————————————————————————————————————————————————————————h2o

METHOD='h2o'

(
python -u run_needle_in_haystack.py --s_len 400 --e_len 16001\
    --model_provider LLaMA3 \
    --model_name /data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct \
    --step 400 \
    --method $METHOD \
    --current_path ${current_path}\
    --model_version qwen2_${METHOD}
) 2>&1  | tee results_needle/logs/qwen2_${METHOD}.log
