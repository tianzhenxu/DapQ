export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5

# model_name='Meta-Llama-3-8B-Instruct'
# model_path='/opt/data/private/sy/tzx/model/LLM-Research/Meta-Llama-3-8B-Instruct'


# model_name='Meta-Llama-3.1-8B-Instruct'
# model_path='/opt/data/private/sy/tzx/model/LLM-Research/Meta-Llama-3.1-8B-Instruct'

# model_name='Mistral-7B-Instruct-v0.2'
# model_path='/opt/data/private/sy/tzx/model/AI-ModelScope/Mistral-7B-Instruct-v0.2'

model_name='Qwen2.5-7B-Instruct'
model_path='/opt/data/private/sy/tzx/model/Qwen/Qwen2.5-7B-Instruct'

# model_name='Qwen3-8B'
# model_path='/opt/data/private/sy/tzx/model/Qwen/Qwen3-8B'



current_path='/opt/data/private/sy/tzx/idea_helmet_result/qwen2/1024'



#——————————————————————————————————————————————————————————————————————————————————————————————————————snapkv
quant='snapkv'


for task in  "longqa" ; do

    output_dir=./results/HELMET/${model_name}/${quant}/${task}
    mkdir -p ${output_dir}    

    python run_helmet.py \
        --config configs/${task}_short.yaml\
        --current_path ${current_path}\
        --model ${model_path} \
        --quant ${quant} \
        --output_path ${output_dir} \

    wait

done



#——————————————————————————————————————————————————————————————————————————————————————————————————————ours
quant='ours'


for task in  "longqa" ; do

    output_dir=./results/HELMET/${model_name}/${quant}/${task}
    mkdir -p ${output_dir}    

    python run_helmet.py \
        --config configs/${task}_short.yaml\
        --current_path ${current_path}\
        --model ${model_path} \
        --quant ${quant} \
        --output_path ${output_dir} \

    wait

done




#——————————————————————————————————————————————————————————————————————————————————————————————————————pyramidkv
quant='pyramidkv'


for task in  "longqa" ; do

    output_dir=./results/HELMET/${model_name}/${quant}/${task}
    mkdir -p ${output_dir}    

    python run_helmet.py \
        --config configs/${task}_short.yaml\
        --current_path ${current_path}\
        --model ${model_path} \
        --quant ${quant} \
        --output_path ${output_dir} \

    wait

done



#——————————————————————————————————————————————————————————————————————————————————————————————————————h2o
quant='h2o'


for task in  "longqa" ; do

    output_dir=./results/HELMET/${model_name}/${quant}/${task}
    mkdir -p ${output_dir}    

    python run_helmet.py \
        --config configs/${task}_short.yaml\
        --current_path ${current_path}\
        --model ${model_path} \
        --quant ${quant} \
        --output_path ${output_dir} \

    wait

done