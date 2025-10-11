# Where Matters More Than What: Decoding aligned KV Cache Compression via Position aware Pseudo queries
### Setup

To install the required packages:

```bash
conda create -n ott python=3.10
conda activate ott
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Then install our CUDA implementation:

```bash
cd quant && pip install -e .
```

### Example

Load model

```python
# LLaMA model with OTT
import torch
import os
from models.llama_OTT import LlamaForCausalLM_OTT
from transformers import LlamaConfig, AutoTokenizer
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.sink_num=SINK_NUM
config.sink_max_size=SINK_MAX_SIZE
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

model = LlamaForCausalLM_OTT.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

# Inference
# e.g., model.generate(...)
```

#### Example

We provide an examples to show how to use OTT. You can check [example.py](./example.py):

```bash
python example.py --[args] [options] 
```
