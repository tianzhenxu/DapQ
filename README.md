# Where Matters More Than What: Decoding_aligned KV Cache Compression via Position_aware Pseudo_queries
### Setup

```python
transformers >= 4.50
flash-attn >= 2.7.4.post1
```

To install the required packages:

```bash
conda create -n DapQ python=3.10
conda activate DaoQ
pip install torch==2.6.0 transformers==4.53.0
pip install --upgrade pip  # enable PEP 660 support
```


## LongBench

We support inference code on `LongBench` to repuduce our result.

Our codebase support Flash Attention v2, Sdpa Attention, etc. The results presented in our paper in based on Flash Attention v2.

自行下载LongBench dataset

## LongBenchV2

We support inference code on `LongBenchV2` to repuduce our result.

Our codebase support Flash Attention v2, Sdpa Attention, etc. The results presented in our paper in based on Flash Attention v2.

自行下载LongBenchV2 dataset

## Ruler

We support inference code on `Ruler` to repuduce our result.

Our codebase support Flash Attention v2, Sdpa Attention, etc. The results presented in our paper in based on Flash Attention v2.

自行下载Ruler dataset

注意：只有qwen3_8B和Meta-Llama-3-8B-Instruct不要使用build chat函数

## Needle in haystack

We support inference code on `Needle in haystack` to repuduce our result.

Our codebase support Flash Attention v2, Sdpa Attention, etc. The results presented in our paper in based on Flash Attention v2.

自行下载Needle_in_haystack dataset

## HELMET

We support inference code on `HELMET` to repuduce our result.

Our codebase support Flash Attention v2, Sdpa Attention, etc. The results presented in our paper in based on Flash Attention v2.

自行下载HELMET dataset
