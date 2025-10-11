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
    string_match_all
)


dataset2maxlen = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "niah_multivalue": 128,
    "cwe": 128,
    "fwe": 128,
    "vt": 128
}

model2maxlen = {
    'llama2-7b-chat': 4096, 
    'Meta-Llama-3-8B-Instruct': 8192, 
    'Llama-2-13b-chat-hf':4096, 
    'Llama-2-7b-chat-hf': 4096,
    'Meta-Llama-3-8B':8192,
    'Meta-Llama-3.1-8B-Instruct':8192,
    'Mistral-7B-Instruct-v0.2':4096,
    'Qwen3-8B':40960,
    'Qwen2.5-7B-Instruct':32768
}


def parse_args():
    parse = argparse.ArgumentParser(description="")
    parse.add_argument("--current_path", type=str, default=None, help="current path")
    parse.add_argument("--model", type=str, default=None, help="model path")
    parse.add_argument("--quant", type=str, default='fp16', help="quant mode")
    parse.add_argument("--output_path", type=str, default=None, help="output path")
    parse.add_argument("--dataset", type=str, default=None, help="dataset")

    args = parse.parse_args()
    return args

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

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



def get_pred(args,model, tokenizer, data, max_length, max_gen, dataset, model_name,out_path):
    preds = []
    
    for i in trange(len(data['train'])):
        
        json_obj=data['train'][i]
        prompt=json_obj["input"] 
        
        prompt = build_chat(tokenizer, prompt, model_name) 
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
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
        output = model.generate(
            input_ids.cuda(),
            max_new_tokens=max_gen,
            past_key_values=past_key_values,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["outputs"]}, f, ensure_ascii=False)
            f.write('\n')

    

if __name__ == "__main__":

    seed_everything(42)
    args = parse_args()
    
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
            
        elif args.quant == 'streamingllm':
            import transformers
            from models.monkeypatch import replace_llama_streamingllm
            replace_llama_streamingllm()
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
            
        elif args.quant == 'streamingllm':
            import transformers
            from models.monkeypatch import replace_qwen2_streamingllm
            replace_qwen2_streamingllm()
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

        elif args.quant == 'streamingllm':
            import transformers
            from models.monkeypatch import replace_qwen3_streamingllm
            replace_qwen3_streamingllm()
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

    data = load_dataset('json', data_files=f"/data/wjk/tzx/RULER/16384/{args.dataset}.jsonl")
    max_length = model2maxlen[os.path.basename(args.model)]
    max_gen = dataset2maxlen[dataset]
    # predict and save
    
    filename = f'{args.output_path}/{args.dataset}.jsonl'
    
    preds = get_pred(args,model, tokenizer, data, max_length, max_gen, dataset, args.model,filename)
    
    # evaluate
    scores = dict()
    
    predictions, answers, lengths = [], [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
    

    score = string_match_all(predictions, answers)
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