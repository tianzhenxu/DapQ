import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer,AutoConfig, AutoModelForCausalLM
import transformers

template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def scorer(predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if prediction is not None:
            pred_lower = prediction.lower()
            truth_lower = ground_truths.lower()
            if pred_lower == truth_lower:
                score = 1.0
        total_score += score
    return total_score,round(100 * total_score / len(predictions), 2)

def parse_args():
    parse = argparse.ArgumentParser(description="")
    parse.add_argument("--current_path", type=str, default=None, help="current path")
    parse.add_argument("--model", type=str, default=None, help="model path")
    parse.add_argument("--quant", type=str, default='fp16', help="quant mode")
    parse.add_argument("--output_path", type=str, default=None, help="output path")
    args = parse.parse_args()
    return args


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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



def get_pred(args,model, tokenizer, data, max_length, max_gen, model_name,out_path):

    for item in tqdm(data):
        context = item['context']
        template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        prompt = build_chat(tokenizer, prompt, model_name)
        
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
        response = pred.strip()
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"response": response,"pred_answer":extract_answer(response), "true_answers": item['answer']}, f, ensure_ascii=False)
            f.write('\n')



if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2maxlen = {'llama2-7b-chat': 4096, 'Meta-Llama-3-8B-Instruct': 8192, 'Llama-2-13b-chat-hf':4096, 'Llama-2-7b-chat-hf': 4096,'Meta-Llama-3-8B':8192,
                    'Meta-Llama-3.1-8B-Instruct':16384,'Mistral-7B-Instruct-v0.2':4096,'Qwen3-8B':16384,'Qwen2.5-7B-Instruct':16384}
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

    data = load_dataset('json', data_files="/data/wjk/tzx/dataset/THUDM/LongBench-v2/data.json")

    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in data['train']]

    max_length = model2maxlen[os.path.basename(args.model)]
    max_gen=128
    filename = f'{args.output_path}/longbench_v2.jsonl'
    preds = get_pred(args,model, tokenizer, data_all, max_length, max_gen, args.model, filename)
    
    # evaluate
    scores = dict()
    
    predictions, answers = [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred_answer"])
            answers.append(data["true_answers"])

    
    total_score,score = scorer(predictions, answers)
    result_str = (
        f'{"="*40}\n'
        f'longbenchv2 results: \n\n'
        f'avg : {score}({int(total_score)}/{len(predictions)})\n'
    )
    # 输出到控制台
    print(result_str)
    # 写入文件（追加模式）
    with open(filename, "a", encoding="utf-8") as f:
        # 写入分隔行
        f.write("\n# Evaluation Results\n")
        # 写入结果字符串
        f.write(result_str)        