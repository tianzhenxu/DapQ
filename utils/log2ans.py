import re
 
def search_gsm8k(args, dataset):
    file_path = f'{args.log_path}/{dataset}.log'
    
    try:
        with open(file_path, 'r') as f:
            read = f.readlines()
            acc_norm = round(float(re.findall(r'\|(.*?)\|', read[-2])[-2])*100, 2)
            acc = round(float(re.findall(r'\|(.*?)\|', read[-3])[-2])*100, 2)
            if acc > acc_norm:
                return acc
            else:
                return acc_norm
    
    except:
        return -1

def search_truthfulqa_gen(args, dataset):
    file_path = f'{args.log_path}/{dataset}.log'
    
    try:
        with open(file_path, 'r') as f:
            read = f.readlines()
            acc = round(float(re.findall(r'\|(.*?)\|', read[-13])[-2])*100, 2)
            return acc
    
    except:
        return -1

def search_coqa(args, dataset):
    file_path = f'{args.log_path}/{dataset}.log'
    
    try:
        with open(file_path, 'r') as f:
            read = f.readlines()
            f1 = round(float(re.findall(r'\|(.*?)\|', read[-2])[-2])*100, 2)
            em = round(float(re.findall(r'\|(.*?)\|', read[-3])[-2])*100, 2)
            return f1, em
    
    except:
        return -1, -1

def longbench(args, dataset):
    if dataset == 'repobench_p':
        dataset = 'repobench-p'
    file_path = f'{args.log_path}/{dataset}.log'
    
    try:
        with open(file_path, 'r') as f:
            read = f.readlines()
            avg = float(read[-1].strip().split(':')[1])
            return avg
    
    except:
        return -1

def search_qasper(args, dataset):
    return longbench(args, dataset)

def search_triviaqa(args, dataset):
    return longbench(args, dataset)
    
def search_trec(args, dataset):
    return longbench(args, dataset)
    
def search_samsum(args, dataset):
    return longbench(args, dataset)
    
def search_lcc(args, dataset):
    return longbench(args, dataset)
    
def search_repobench_p(args, dataset):
    return longbench(args, dataset)

def search_multi_news(args, dataset):
    return longbench(args, dataset)
    
def search_multifieldqa_en(args, dataset):
    return longbench(args, dataset)
    
def search_hotpotqa(args, dataset):
    return longbench(args, dataset)
    
def search_2wikimqa(args, dataset):
    return longbench(args, dataset)
    
def search_gov_report(args, dataset):
    return longbench(args, dataset)
    
def search_passage_count(args, dataset):
    return longbench(args, dataset)
    
def search_passage_retrieval_en(args, dataset):
    return longbench(args, dataset)