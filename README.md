# NanoQuantization

A self-contained end-to-end quantization framework that currently supports AWQ (Activation-aware Weight Quantization) and Qwen3 models.

* [Companion blog](https://liyuan24.github.io/writings/2026_01_06_post_training_quantization.html)

## Introduction

This repository provides a complete quantization pipeline from model loading to quantized model deployment. The implementation intentionally avoids using the Hugging Face Transformers library to keep the quantization code concise. Instead, models are implemented layer by layer, providing full control over the quantization process and making it easier to understand and modify the quantization logic.

# Perplexity Benchmark

## Results

| PPL | No Quantization | AWQ | AWQ with no clipping|
|-------|------|---------------|------|
| Qwen3-0.6B | 15.38 | 19.75| 20.38|

The results are obtained by running the following script:

```bash
python -m scripts.qwen3_0pt6_billion \
    --quantize_model \
    --model_path /workspace/huggingface/Qwen3-0.6B \
    --output_dir ./quantized_models/qwen3-0.6b-awq \
    --eval_ppl
```

# Push the quantized model to Hugging Face Hub

```bash
python -m scripts.qwen3_0pt6_billion \
    --output_dir ./quantized_models/qwen3-0.6b-awq \
    --save_quantized_model \
    --push_to_hub \
    --hub_model_id <repo_id/model_name>
```

# Load the quantized model from Hugging Face

For Qwen3-0.6B quantized model, I have uploaded it to Hugging Face [seangogo/qwen3_0pt6b_awq](https://huggingface.co/seangogo/qwen3_0pt6b_awq)

To load the model,

First, download the model from Hugging Face

```bash
hf download seangogo/qwen3_0pt6b_awq \
  --local-dir <local_path>
```

Then run the following script to load the model

```bash
python -m scripts.load_quantized_model --local_path <local_path>
```

