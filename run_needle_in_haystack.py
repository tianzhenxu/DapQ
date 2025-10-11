"""
This script is adapted from 
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import os 
import glob

import json
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import numpy as np
import argparse
from rouge_score import rouge_scorer
import transformers
import sys
import os
from models.monkeypatch import replace_llama_cache_ours,replace_qwen2_cache_ours,replace_qwen3_cache_ours,replace_mistral_cache_ours
from datetime import datetime, timezone
import time
import torch

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",#    \n在旧金山做的最棒的是事是吃一个三明治以及在晴天坐在多洛雷斯公园里\n
                 haystack_dir="data/PaulGrahamEssays", # PaulGrahamEssays  
                 retrieval_question="The best thing to do in San Francisco is: ", 
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 model_name='',
                 model_version=None, 
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True, 
                 current_path=None,
                 step=100, 
                 method=None):
        """        
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = method
        self.current_path=current_path

        self.model_version = model_version

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name
            
        config =AutoConfig.from_pretrained(self.model_name)
        self.enc=AutoTokenizer.from_pretrained(self.model_name, use_fast=False,trust_remote_code=True)
        import sys
        sys.path.append(self.current_path) 
        if 'llama' in self.model_name.lower():
            if self.method == 'ours':
                import transformers
                from models.monkeypatch import replace_llama_ours,replace_llama_cache_ours
                replace_llama_ours()
                
            elif self.method == 'snapkv':
                import transformers
                from models.monkeypatch import replace_llama_snapkv
                replace_llama_snapkv()
                config._attn_implementation="flash_attention_2"
                
            elif self.method == 'pyramidkv':
                import transformers
                from models.monkeypatch import replace_llama_pyramidkv
                replace_llama_pyramidkv()
                config._attn_implementation="flash_attention_2"            
                
            elif self.method == 'h2o':
                import transformers
                from models.monkeypatch import replace_llama_h2o
                replace_llama_h2o()
                config._attn_implementation="flash_attention_2"     
                
            elif self.method == 'fp16':
                pass
            
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )      
        elif 'mistral' in self.model_name.lower():
            if self.method == 'ours':
                import transformers
                from models.monkeypatch import replace_mistral_ours,replace_mistral_cache_ours
                replace_mistral_ours()
                
            elif self.method == 'snapkv':
                import transformers
                from models.monkeypatch import replace_mistral_snapkv
                replace_mistral_snapkv()
                config._attn_implementation="flash_attention_2"
                
            elif self.method == 'fp16':
                pass
            
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )      
        elif 'qwen2.5' in self.model_name.lower():
            if self.method == 'ours':
                import transformers
                from models.monkeypatch import replace_qwen2_ours,replace_qwen2_cache_ours
                replace_qwen2_ours()
                
            elif self.method == 'snapkv':
                import transformers
                from models.monkeypatch import replace_qwen2_snapkv
                replace_qwen2_snapkv()
                config._attn_implementation="flash_attention_2"

            elif self.method == 'pyramidkv':
                import transformers
                from models.monkeypatch import replace_qwen2_pyramidkv
                replace_qwen2_pyramidkv()
                config._attn_implementation="flash_attention_2"   
                
            elif self.method == 'h2o':
                import transformers
                from models.monkeypatch import replace_qwen2_h2o
                replace_qwen2_h2o()
                config._attn_implementation="flash_attention_2"      
                        
            elif self.method == 'fp16':
                pass
            
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )      
        elif 'qwen3' in self.model_name.lower():
            if self.method == 'ours':
                import transformers
                from models.monkeypatch import replace_qwen3_ours,replace_qwen3_cache_ours
                replace_qwen3_ours()
                
            elif self.method == 'snapkv':
                import transformers
                from models.monkeypatch import replace_qwen3_snapkv
                replace_qwen3_snapkv()
                config._attn_implementation="flash_attention_2"

            elif self.method == 'pyramidkv':
                import transformers
                from models.monkeypatch import replace_qwen3_pyramidkv
                replace_qwen3_pyramidkv()
                config._attn_implementation="flash_attention_2"   

            elif self.method == 'h2o':
                import transformers
                from models.monkeypatch import replace_qwen3_h2o
                replace_qwen3_h2o()
                config._attn_implementation="flash_attention_2"   
                
                
            elif self.method == 'fp16':
                pass
            
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )      
        else:
            raise NotImplementedError

        self.model_to_test.eval()
        
    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure

        test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        return test_format
    
    def build_chat(self,tokenizer, prompt, model_name):
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


    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        # prompt = self.build_chat(self.enc, prompt, self.model_name)
        test_start_time = time.time()
        

        prompt = self.enc(prompt, return_tensors="pt")
        input_ids = prompt['input_ids'].to(self.model_to_test.device)
        
        past_key_values=None
        
        if self.method == 'ours':
            add = torch.cat((input_ids[0, :4], input_ids[0, -28:]), dim=0).unsqueeze(0)      
                
            if 'llama' in self.model_name.lower():
                Cache=replace_llama_cache_ours()             
            elif 'mistral' in self.model_name.lower():
                Cache=replace_mistral_cache_ours()                 
            elif 'qwen2.5' in self.model_name.lower():
                Cache=replace_qwen2_cache_ours()                    
            elif 'qwen3' in self.model_name.lower():
                Cache=replace_qwen3_cache_ours()
            else:    
                raise NotImplementedError
            past_key_values = Cache(fake_size=32,cache_budget=256,kernel_size=5)
            input_ids=torch.cat([input_ids,add],dim=-1)              
                
            
        output_ids = self.model_to_test.generate(
            input_ids, 
            output_attentions=False,
            past_key_values=past_key_values,
            max_new_tokens=30,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]]
        )
        response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        
        print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        if len(response) != 0:
            score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
        else:
            score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('results_needle/contexts'):
                os.makedirs('results_needle/contexts')

            if not os.path.exists(f'results_needle/contexts/{self.model_version}'):
                os.makedirs(f'results_needle/contexts/{self.model_version}')

            with open(f'results_needle/contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results_needle/results'):
                os.makedirs('results_needle/results')
            
            if not os.path.exists(f'results_needle/results/{self.model_version}'):
                os.makedirs(f'results_needle/results/{self.model_version}')

            # Save the result to file for retesting
            p = f'results_needle/results/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results_needle/results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(text, add_special_tokens=False)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context, add_special_tokens=False))


    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context, add_special_tokens=False)

        
    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length], skip_special_tokens=True)


    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_version', type=str, default=None, help='provider of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument("--current_path", type=str, default=None, help="current path")
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--method', type=str, default='fp16')
    args = parser.parse_args()

    

    ht = LLMNeedleHaystackTester(model_name=args.model_name, 
                                 model_provider=args.model_provider,
                                 model_version=args.model_version, 
                                 context_lengths_min=args.s_len,
                                 save_contexts=True,
                                 save_results=True,
                                 context_lengths_max=args.e_len, 
                                 step=args.step, 
                                 method=args.method, 
                                 current_path=args.current_path
                                 )

    ht.start_test(args)