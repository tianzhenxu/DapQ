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
