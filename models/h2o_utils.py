import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    # def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups): # true
        
    #     # check if prefix phase
    #     assert key_states.shape[-2] == query_states.shape[-2]
    #     bsz, num_heads, q_len, head_dim = query_states.shape
    
        
    #     if q_len < self.max_capacity_prompt:
    #         return key_states, value_states
    #     else:
    #         k_len = key_states.shape[-2]
    #         attn_cache = torch.zeros(bsz, num_heads, k_len, device=query_states.device, dtype=query_states.dtype)
    #         # 设置批处理大小，可根据显存情况调整
    #         batch_size = 256 
    #         num_batches = (q_len + batch_size - 1) // batch_size  # 计算总批次数
            
    #         for batch_idx in range(num_batches):
    #             # 计算当前批次的起始和结束索引
    #             start = batch_idx * batch_size
    #             end = min((batch_idx + 1) * batch_size, q_len)
    #             batch_q_len = end - start  # 当前批次实际query数量
                
    #             # 1. 获取当前批次的query（形状：[1, 32, batch_q_len, 128]）
    #             q_batch = query_states[:, :, start:end, :]
                
    #             # 2. 计算与所有key的原始相似度（未归一化）
    #             # 结果形状：[1, 32, batch_q_len, k_len]
    #             raw_weights = torch.matmul(q_batch, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                
    #             # 3. 创建批量因果掩码：每个位置(i,j)表示第i条query是否能看到第j条key
    #             # 生成query索引范围 [start, start+1, ..., end-1]
    #             query_indices = torch.arange(start, end, device=query_states.device).view(1, 1, -1, 1)
    #             # 生成key索引：[0, 1, ..., k_len-1]
    #             key_indices = torch.arange(k_len, device=query_states.device).view(1, 1, 1, -1)
    #             # 掩码规则：key索引 > query索引 的位置需要被掩码
    #             mask = (key_indices > query_indices)  # 形状：[1, 1, batch_q_len, k_len]
    #             # 应用掩码（将需要屏蔽的位置设为负无穷）
    #             masked_weights = raw_weights.masked_fill(mask, torch.finfo(raw_weights.dtype).min)

    #             # 4. 对掩码后的权重做softmax（按key维度归一化）
    #             attn_weights = nn.functional.softmax(masked_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
    #             # 5. 累加当前批次的注意力权重到缓存
    #             attn_weights_sum = attn_weights.sum(dim=-2)  # 形状：[1, 32, k_len]
    #             attn_cache += attn_weights_sum  # 累加到缓存中

            
            
    #         indices = attn_cache[:,:,: -self.window_size].topk(self.max_capacity_prompt - self.window_size, dim=-1).indices  # 形状：[1, 32, topk]
    #         indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # 扩展维度用于gather
    #         k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
    #         v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
    #         k_cur = key_states[:, :, -self.window_size:, :]
    #         v_cur = value_states[:, :, -self.window_size:, :]
    #         key_states = torch.cat([k_past_compress, k_cur], dim = 2)
    #         value_states = torch.cat([v_past_compress, v_cur], dim = 2)
    #         return key_states, value_states
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups): # 有recent
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
    
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            recent_size=int(self.max_capacity_prompt/2-self.window_size)
            
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights.sum(dim = -2)


            attn_cache=attn_weights_sum[:, :, : -(self.window_size+recent_size)]
            
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size-recent_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)                        
            
            k_past_compress = key_states[:, :, :-(self.window_size+recent_size), :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-(self.window_size+recent_size), :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -(self.window_size+recent_size):, :]
            v_cur = value_states[:, :, -(self.window_size+recent_size):, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)     
                       
            return key_states, value_states







def init_h2o(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 256
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        )
