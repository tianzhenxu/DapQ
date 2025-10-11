export HF_ENDPOINT=https://hf-mirror.com
declare -A path_map
# path_map['llama2-7b-chat-hf']='/data/wjk/llama2/Llama-2-7b-chat-hf'
# path_map['Meta-Llama-3-8B']='/opt/data/private/sy/tzx/model/LLM-Research/Meta-Llama-3-8B'
# -------------Edit here-------------
export CUDA_VISIBLE_DEVICES=1
# model=$1
# model_name='Meta-Llama-3-8B-Instruct'
# model_path='/data/wjk/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct'

# model_name='Mistral-7B-Instruct-v0.2'
# model_path='/data/wjk/tzx/model/AI-ModelScope/Mistral-7B-Instruct-v0.2'

model_name='Qwen2.5-7B-Instruct'
model_path='/data/wjk/tzx/model/Qwen/Qwen2.5-7B-Instruct'

# model_name='Qwen3-8B'
# model_path='/data/wjk/tzx/model/Qwen/Qwen3-8B'


current_path='/data/wjk/tzx/idea_all_model'

# quant=$2
# quant='snapkv'
quant='ours'
# quant='fp16'
# -------------Edit here-------------

eval_e='true'
# eval_e='False'

if [ "$eval_e" = "true" ]; then
    # eval in longbench-E
    declare -a longbench_list=(
        "qasper"
        "multifieldqa_en"
        "hotpotqa"
        "2wikimqa"
        "gov_report"
        "multi_news"
        "trec"
        "triviaqa"
        "samsum"
        "lcc"
        "repobench-p"
        "passage_retrieval_en"
        "passage_count"
    )
else
    # eval in longbench
    declare -a longbench_list=(
        "narrativeqa"
        "qasper"
        "multifieldqa_en"
        "hotpotqa"
        "2wikimqa"
        "musique"
        "gov_report"
        "multi_news"
        "qmsum"
        "trec"
        "triviaqa"
        "samsum"
        "lcc"
        "repobench-p"
        "passage_retrieval_en"
        "passage_count"
    )
fi



for dataset in "${longbench_list[@]}"; do

    if [ "$eval_e" = "true" ]; then
        output_dir=./results/Longbench-E/${model_name}/${quant}/${dataset}
        mkdir -p ${output_dir}    

        python ./run_longbench.py \
            --current_path ${current_path}\
            --model ${model_path} \
            --quant ${quant} \
            --output_path ${output_dir} \
            --dataset ${dataset} \
            --eval_e

    else
        output_dir=./results/Longbench/${model_name}/${quant}/${dataset}
        mkdir -p ${output_dir}    

        python ./run_longbench.py \
            --current_path ${current_path}\
            --model ${model_path} \
            --quant ${quant} \
            --output_path ${output_dir} \
            --dataset ${dataset} \
            
    fi
    wait

done
