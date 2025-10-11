import transformers
from .llama_ours import OursCache as OursCache_llama_ours,LlamaForCausalLM_foward_ours,LlamaAttention_foward_ours, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_ours
from .llama_snapkv import LlamaForCausalLM_foward_snapkv,llama_flash_attn2_forward as llama_flash_attn2_forward_snapkv, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_snapkv
from .llama_pyramidkv import LlamaForCausalLM_foward_pyramidkv,llama_flash_attn2_forward as llama_flash_attn2_forward_pyramidkv, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_pyramidkv
from .llama_h2o import LlamaForCausalLM_foward_h2o,llama_flash_attn2_forward as llama_flash_attn2_forward_h2o, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_h2o
from .llama_streamingllm import LlamaForCausalLM_foward_streamingllm,llama_flash_attn2_forward as llama_flash_attn2_forward_streamingllm, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_streamingllm


from .mistral_ours import OursCache as OursCache_mistral_ours,MistralForCausalLM_foward_ours,MistralAttention_foward_ours, prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_ours
from .mistral_snapkv import MistralForCausalLM_foward_snapkv,mistral_flash_attn2_forward as mistral_flash_attn2_forward_snapkv, prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_snapkv

from .qwen2_ours import OursCache as OursCache_qwen2_ours,Qwen2ForCausalLM_foward_ours,Qwen2Attention_foward_ours, prepare_inputs_for_generation_qwen2 as prepare_inputs_for_generation_qwen2_ours
from .qwen2_snapkv import Qwen2ForCausalLM_foward_snapkv,qwen2_flash_attn2_forward as qwen2_flash_attn2_forward_snapkv, prepare_inputs_for_generation_qwen2 as prepare_inputs_for_generation_qwen2_snapkv
from .qwen2_pyramidkv import Qwen2ForCausalLM_foward_pyramidkv,qwen2_flash_attn2_forward as qwen2_flash_attn2_forward_pyramidkv, prepare_inputs_for_generation_qwen2 as prepare_inputs_for_generation_qwen2_pyramidkv
from .qwen2_h2o import Qwen2ForCausalLM_foward_h2o,qwen2_flash_attn2_forward as qwen2_flash_attn2_forward_h2o, prepare_inputs_for_generation_qwen2 as prepare_inputs_for_generation_qwen2_h2o
from .qwen2_streamingllm import Qwen2ForCausalLM_foward_streamingllm,qwen2_flash_attn2_forward as qwen2_flash_attn2_forward_streamingllm, prepare_inputs_for_generation_qwen2 as prepare_inputs_for_generation_qwen2_streamingllm

from .qwen3_ours import OursCache as OursCache_qwen3_ours,Qwen3ForCausalLM_foward_ours,Qwen3Attention_foward_ours, prepare_inputs_for_generation_qwen3 as prepare_inputs_for_generation_qwen3_ours
from .qwen3_snapkv import Qwen3ForCausalLM_foward_snapkv,qwen3_flash_attn2_forward as qwen3_flash_attn2_forward_snapkv, prepare_inputs_for_generation_qwen3 as prepare_inputs_for_generation_qwen3_snapkv
from .qwen3_pyramidkv import Qwen3ForCausalLM_foward_pyramidkv,qwen3_flash_attn2_forward as qwen3_flash_attn2_forward_pyramidkv, prepare_inputs_for_generation_qwen3 as prepare_inputs_for_generation_qwen3_pyramidkv
from .qwen3_h2o import Qwen3ForCausalLM_foward_h2o,qwen3_flash_attn2_forward as qwen3_flash_attn2_forward_h2o, prepare_inputs_for_generation_qwen3 as prepare_inputs_for_generation_qwen3_h2o
from .qwen3_streamingllm import Qwen3ForCausalLM_foward_streamingllm,qwen3_flash_attn2_forward as qwen3_flash_attn2_forward_streamingllm, prepare_inputs_for_generation_qwen3 as prepare_inputs_for_generation_qwen3_streamingllm

"""-------------------------------------------------llama-------------------------------------------------"""
def replace_llama_streamingllm():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_streamingllm
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_foward_streamingllm
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_streamingllm


def replace_llama_h2o():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_h2o
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_foward_h2o
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_h2o

def replace_llama_pyramidkv():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_pyramidkv
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_foward_pyramidkv
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_pyramidkv

def replace_llama_snapkv():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_snapkv
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_foward_snapkv
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_snapkv

def replace_llama_ours():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_ours
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_foward_ours
    transformers.models.llama.modeling_llama.LlamaAttention.forward = LlamaAttention_foward_ours
    
def replace_llama_cache_ours():
    return OursCache_llama_ours
"""-------------------------------------------------mistral-------------------------------------------------"""
def replace_mistral_snapkv():
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_snapkv
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.forward = MistralForCausalLM_foward_snapkv
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_snapkv

def replace_mistral_ours():
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_ours
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.forward = MistralForCausalLM_foward_ours
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = MistralAttention_foward_ours
    
def replace_mistral_cache_ours():
    return OursCache_mistral_ours
"""-------------------------------------------------qwen2-------------------------------------------------"""
def replace_qwen2_streamingllm():
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2_streamingllm
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM_foward_streamingllm 
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_streamingllm
    

def replace_qwen2_h2o():
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2_h2o
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM_foward_h2o 
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_h2o
    
def replace_qwen2_pyramidkv():
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2_pyramidkv
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM_foward_pyramidkv
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_pyramidkv

def replace_qwen2_snapkv():
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2_snapkv
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM_foward_snapkv
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_snapkv

def replace_qwen2_ours():
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2_ours
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM_foward_ours
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_foward_ours
    
def replace_qwen2_cache_ours():
    return OursCache_qwen2_ours
"""-------------------------------------------------qwen3-------------------------------------------------"""
def replace_qwen3_streamingllm():
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3_streamingllm
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3ForCausalLM_foward_streamingllm
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward =  qwen3_flash_attn2_forward_streamingllm

def replace_qwen3_h2o():
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3_h2o
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3ForCausalLM_foward_h2o
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward =  qwen3_flash_attn2_forward_h2o
    
def replace_qwen3_pyramidkv():
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3_pyramidkv
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3ForCausalLM_foward_pyramidkv
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_flash_attn2_forward_pyramidkv

def replace_qwen3_snapkv():
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3_snapkv
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3ForCausalLM_foward_snapkv
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_flash_attn2_forward_snapkv

def replace_qwen3_ours():
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3_ours
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3ForCausalLM_foward_ours
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = Qwen3Attention_foward_ours
    
def replace_qwen3_cache_ours():
    return OursCache_qwen3_ours