import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from tqdm import trange

import transformers
from transformers import AutoTokenizer,AutoConfig, AutoModelForCausalLM

import argparse

from utils.metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parse = argparse.ArgumentParser(description="")
    parse.add_argument("--current_path", type=str, default=None, help="current path")
    parse.add_argument("--model", type=str, default=None, help="model path")
    parse.add_argument("--quant", type=str, default='fp16', help="quant mode")
    parse.add_argument("--output_path", type=str, default=None, help="output path")
    parse.add_argument("--dataset", type=str, default=None, help="dataset")
    parse.add_argument('--eval_e', action='store_true', help="choose to eval in longbench-e")

    args = parse.parse_args()
    return args

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            # import pdb; pdb.set_trace()
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # import pdb; pdb.set_trace()
    elif "mistral-7b-instruct" in model_name.lower() or "qwen2" in model_name.lower():
        messages = [
            {"role": "user","content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "qwen3" in model_name.lower():
        messages = [
            {"role": "user","content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages , tokenize=False, add_generation_prompt=True,enable_thinking=False)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(args,model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name,out_path):
    preds = []
    
    for i in trange(len(data['train'])):
        
        json_obj=data['train'][i]
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device).input_ids

        past_key_values=None

        if args.quant == 'ours':
            add = torch.cat((input_ids[0, :4], input_ids[0, -28:]), dim=0).unsqueeze(0)      
                  
            if 'llama' in args.model.lower():
                Cache=replace_llama_cache_ours()             
            elif 'mistral' in args.model.lower():
                Cache=replace_mistral_cache_ours()                 
            elif 'qwen2.5' in args.model.lower():
                Cache=replace_qwen2_cache_ours()                    
            elif 'qwen3' in args.model.lower():
                Cache=replace_qwen3_cache_ours()
            else:    
                raise NotImplementedError
            past_key_values = Cache(fake_size=32,cache_budget=512,kernel_size=5)
            input_ids=torch.cat([input_ids,add],dim=-1)                   

            
        context_length = input_ids.shape[-1]    
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                input_ids.cuda(),
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values=past_key_values,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                input_ids.cuda(),
                max_new_tokens=max_gen,
                past_key_values=past_key_values,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    dataset2maxlen = json.load(open("./config/longbench/dataset2maxlen.json", "r"))
    dataset2prompt = json.load(open("./config/longbench/dataset2prompt.json", "r"))
    model2maxlen = {'llama2-7b-chat': 4096, 'Meta-Llama-3-8B-Instruct': 8192, 'Llama-2-13b-chat-hf':4096, 'Llama-2-7b-chat-hf': 4096,'Meta-Llama-3-8B':8192,
                    'Meta-Llama-3.1-8B-Instruct':131072,'Mistral-7B-Instruct-v0.2':4096,'Qwen3-8B':8192,'Qwen2.5-7B-Instruct':8192}
    config =AutoConfig.from_pretrained(args.model)
    tokenizer=AutoTokenizer.from_pretrained(args.model, use_fast=False,trust_remote_code=True)
    import sys
    sys.path.append(args.current_path) 
    if 'llama' in args.model.lower():
        if args.quant == 'ours':
            import transformers
            from models.monkeypatch import replace_llama_ours,replace_llama_cache_ours
            replace_llama_ours()
            
        elif args.quant == 'snapkv':
            import transformers
            from models.monkeypatch import replace_llama_snapkv
            replace_llama_snapkv()
            config._attn_implementation="flash_attention_2"
            
        elif args.quant == 'pyramidkv':
            import transformers
            from models.monkeypatch import replace_llama_pyramidkv
            replace_llama_pyramidkv()
            config._attn_implementation="flash_attention_2"            
            
        elif args.quant == 'h2o':
            import transformers
            from models.monkeypatch import replace_llama_h2o
            replace_llama_h2o()
            config._attn_implementation="flash_attention_2"     
            
        elif args.quant == 'fp16':
            pass
        
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    elif 'mistral' in args.model.lower():
        if args.quant == 'ours':
            import transformers
            from models.monkeypatch import replace_mistral_ours,replace_mistral_cache_ours
            replace_mistral_ours()
            
        elif args.quant == 'snapkv':
            import transformers
            from models.monkeypatch import replace_mistral_snapkv
            replace_mistral_snapkv()
            config._attn_implementation="flash_attention_2"
            
        elif args.quant == 'fp16':
            pass
        
        model = transformers.MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    elif 'qwen2.5' in args.model.lower():
        if args.quant == 'ours':
            import transformers
            from models.monkeypatch import replace_qwen2_ours,replace_qwen2_cache_ours
            replace_qwen2_ours()
            
        elif args.quant == 'snapkv':
            import transformers
            from models.monkeypatch import replace_qwen2_snapkv
            replace_qwen2_snapkv()
            config._attn_implementation="flash_attention_2"

        elif args.quant == 'pyramidkv':
            import transformers
            from models.monkeypatch import replace_qwen2_pyramidkv
            replace_qwen2_pyramidkv()
            config._attn_implementation="flash_attention_2"   
            
        elif args.quant == 'h2o':
            import transformers
            from models.monkeypatch import replace_qwen2_h2o
            replace_qwen2_h2o()
            config._attn_implementation="flash_attention_2"      
                     
        elif args.quant == 'fp16':
            pass
        
        model = transformers.Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    elif 'qwen3' in args.model.lower():
        if args.quant == 'ours':
            import transformers
            from models.monkeypatch import replace_qwen3_ours,replace_qwen3_cache_ours
            replace_qwen3_ours()
            
        elif args.quant == 'snapkv':
            import transformers
            from models.monkeypatch import replace_qwen3_snapkv
            replace_qwen3_snapkv()
            config._attn_implementation="flash_attention_2"

        elif args.quant == 'pyramidkv':
            import transformers
            from models.monkeypatch import replace_qwen3_pyramidkv
            replace_qwen3_pyramidkv()
            config._attn_implementation="flash_attention_2"   

        elif args.quant == 'h2o':
            import transformers
            from models.monkeypatch import replace_qwen3_h2o
            replace_qwen3_h2o()
            config._attn_implementation="flash_attention_2"   
            
            
        elif args.quant == 'fp16':
            pass
        
        model = transformers.Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    else:
        raise NotImplementedError

    model.eval()

    # load dataset    
    dataset = args.dataset

    #data = load_dataset('/data/wjk/tzx/THUDM/LongBench/', f"{dataset}_e", split='test')
    if args.eval_e:
        data = load_dataset('json', data_files=f"/data/wjk/tzx/THUDM/LongBench/data/{dataset}_e.jsonl")
    else:
        data = load_dataset('json', data_files=f"/data/wjk/tzx/THUDM/LongBench/data/{dataset}.jsonl")
    prompt_format = dataset2prompt[dataset]
    max_length = model2maxlen[os.path.basename(args.model)]
    max_gen = dataset2maxlen[dataset]
    # predict and save
    filename = f'{args.output_path}/{args.dataset}.jsonl'
    preds = get_pred(args,model, tokenizer, data, max_length, max_gen, prompt_format, dataset, args.model,filename)
    
    # evaluate
    scores = dict()
    
    predictions, answers, lengths = [], [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
    
    
    if args.eval_e:           
        score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
        average_score = round((score["0-4k"] + score["4-8k"] + score["8k+"])/3, 2)
        # 构建结果字符串
        result_str = (
            f'{"="*40}\n'
            f'{str(args.dataset)} results: \n\n'
            f'0-4k: {score["0-4k"]}\n'
            f'4-8k: {score["4-8k"]}\n'
            f'8k+ : {score["8k+"]}\n'
            f'avg : {average_score}\n'
        )
        # 输出到控制台
        print(result_str)
        # 写入文件（追加模式）
        with open(filename, "a", encoding="utf-8") as f:
            # 写入分隔行
            f.write("\n# Evaluation Results\n")
            # 写入结果字符串
            f.write(result_str)
            
    else:
        score = scorer(dataset, predictions, answers, all_classes)
        result_str = (
            f'{"="*40}\n'
            f'{str(args.dataset)} results: \n\n'
            f'avg : {score}\n'
        )
        # 输出到控制台
        print(result_str)
        # 写入文件（追加模式）
        with open(filename, "a", encoding="utf-8") as f:
            # 写入分隔行
            f.write("\n# Evaluation Results\n")
            # 写入结果字符串
            f.write(result_str)        