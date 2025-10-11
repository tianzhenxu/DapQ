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
from transformers.cache_utils import DynamicCache
import argparse
from models.kv_cache import LaCache
from utils.metrics import (
    string_match_all
)


dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
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

os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parse = argparse.ArgumentParser(description="")
    parse.add_argument("--current_path", type=str, default=None, help="current path")
    parse.add_argument("--model", type=str, default=None, help="model path")
    parse.add_argument("--quant", type=str, default='fp16', help="quant mode")
    parse.add_argument("--output_path", type=str, default=None, help="output path")
    parse.add_argument("--dataset", type=str, default=None, help="dataset")
    parse.add_argument("--span", type=int, default=16)
    parse.add_argument("--overlap", type=int, default=0)
    parse.add_argument('--budget', type=float, default=0.5)
    
    args = parse.parse_args()
    return args



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




def greedy_generate(model, tokenizer, input_ids, past_key_values, max_length, max_gen_len, eos_token_id=None, 
                    method=None,budget=None, span=None, overlap=None):
    with torch.no_grad():
        past_key_values = DynamicCache()
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True
        )
        # import pdb;pdb.set_trace()
        outputs = model(**inputs)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]
    
        if method == "lacache":
            # cache_size = int(budget * max_length)
            # start_size = min(128, max(1, past_key_values[0][0].shape[2] - (len(past_key_values) + 1)), cache_size - (len(past_key_values) + 1))
            kv_cache = LaCache(cache_size=256, start_size=4, span=span, overlap=overlap)
            legacy_cache = kv_cache(past_key_values.to_legacy_cache())
            past_key_values = DynamicCache.from_legacy_cache(legacy_cache)
            # import pdb;pdb.set_trace()
    # 初始化 decoding 阶段的变量
    current_position = input_ids.shape[1]  # 初始 cache_position 为 input_ids 的长度
    device = input_ids.device  # 获取设备信息
    with torch.no_grad():
        for _ in range(max_gen_len - 1):
            cache_position = torch.tensor([current_position], device=device)
            inputs = model.prepare_inputs_for_generation(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True
            )
            outputs = model(**inputs)
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            current_position += 1
            generated_ids.append(pred_token_idx.item())
            if pred_token_idx in eos_token_id:
                break
        
    generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
    generated_text = " ".join(generated_text[:])
    return generated_text


def get_pred(args,model, tokenizer, data, max_length, max_gen, model_name,out_path):
    preds = []
    
    for i in trange(len(data['train'])):
        
        json_obj=data['train'][i]
        prompt=json_obj["input"] 
        
        prompt = build_chat(tokenizer, prompt, model_name)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device).input_ids
        
        
        pred = greedy_generate(model, tokenizer, input_ids, None, max_length, max_gen, [tokenizer.eos_token_id], 
                                   method=args.quant,budget=args.budget, span=args.span, overlap=args.overlap)              

        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["outputs"]}, f, ensure_ascii=False)
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
    config =AutoConfig.from_pretrained(args.model)
    tokenizer=AutoTokenizer.from_pretrained(args.model, use_fast=False,trust_remote_code=True)
    import sys
    sys.path.append(args.current_path) 
    
    import sys
    sys.path.append(args.current_path) 
    if 'llama' in args.model.lower():
        if args.quant == 'lacache':
            import transformers
            from models.llama_patch import replace_llama_lacache
            replace_llama_lacache()
            config._attn_implementation="flash_attention_2"
        
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    elif 'mistral' in args.model.lower():
        if args.quant == 'lacache':
            pass
        
        model = transformers.MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    elif 'qwen2.5' in args.model.lower():
        if args.quant == 'lacache':
            import transformers
            from models.qwen2_patch import replace_qwen2_lacache
            replace_qwen2_lacache()
            config._attn_implementation="flash_attention_2"
            
        model = transformers.Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )      
    elif 'qwen3' in args.model.lower():
        if args.quant == 'lacache':
            import transformers
            from models.qwen3_patch import replace_qwen3_lacache
            replace_qwen3_lacache()
            config._attn_implementation="flash_attention_2"
        
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

    data = load_dataset('json', data_files=f"/data/wjk/tzx/RULER/8192/{args.dataset}.jsonl")
    max_length = model2maxlen[os.path.basename(args.model)]
    max_gen = dataset2maxlen[dataset]
    # predict and save
    
    filename = f'{args.output_path}/{args.dataset}.jsonl'
    
    preds = get_pred(args,model, tokenizer, data, max_length, max_gen, args.model,filename)
    
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