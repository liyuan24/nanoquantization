#!/usr/bin/env python3
"""
Quantization script for Qwen3-0.6B model using AWQ (Activation-aware Weight Quantization).

This script:
1. Loads the Qwen3-0.6B model from Hugging Face
2. Applies AWQ 4-bit quantization with group_size=128
3. Saves the quantized model in Hugging Face safetensors format
4. Optionally evaluates perplexity before and after quantization

The quantized model is saved in a format compatible with Hugging Face Transformers,
using safetensors for efficient and safe weight storage.

Usage:
    python -m scripts.qwen3_0pt6_billion \
    --model_path /workspace/huggingface/Qwen3-0.6B \
    --output_dir ./quantized_models/qwen3-0.6b-awq \
    --eval_ppl
"""

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import save_file

from nanoquantization.awq.quantizer import AWQQuantizer
from nanoquantization.awq.models.qwen3 import Qwen3ForCausalLM
from nanoquantization.utils.loader import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Qwen3-0.6B model using AWQ")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Path to the pre-trained model (local path or HF model ID)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_models/qwen3-0.6b-awq",
        help="Directory to save the quantized model",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        default=4,
        help="Number of bits for weight quantization (default: 4)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)",
    )
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        default="wikitext",
        help="Dataset for calibration (default: wikitext)",
    )
    parser.add_argument(
        "--calibration_subset",
        type=str,
        default="wikitext-103-v1",
        help="Subset of calibration dataset (default: wikitext-103-v1)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length for calibration (default: 512)",
    )
    parser.add_argument(
        "--no_clip",
        action="store_false",
        dest="apply_clip",
        help="Disable clipping (default: enabled)",
    )
    parser.add_argument(
        "--no_zero_point",
        action="store_false",
        dest="zero_point",
        help="Disable zero point in quantization (default: enabled)",
    )
    parser.add_argument(
        "--eval_ppl",
        action="store_true",
        help="Evaluate perplexity before and after quantization",
    )
    parser.add_argument(
        "--save_quantized_model",
        action="store_true",
        help="Save the quantized model to disk",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the quantized model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for Hugging Face Hub (e.g., 'username/model-name'). If not specified, uses output_dir basename",
    )
    return parser.parse_args()


def load_qwen3_model(model_path: str):
    """Load Qwen3 model and tokenizer."""
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Set default dtype and device
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")
    
    # Initialize model with config
    model = Qwen3ForCausalLM(
        num_hidden_layers=hf_config.num_hidden_layers,
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        total_num_heads=hf_config.num_attention_heads,
        total_num_kv_heads=hf_config.num_key_value_heads,
        max_position_embeddings=hf_config.max_position_embeddings,
        intermediate_size=hf_config.intermediate_size,
        head_dim=hf_config.head_dim,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
    )
    
    # Load weights
    print("Loading model weights...")
    load_model(model, model_path)
    print("✓ Model loaded successfully\n")
    
    return model, tokenizer, hf_config


def save_quantized_model(model, tokenizer, hf_config, output_dir: str, args):
    """Save the quantized model, tokenizer, and config in safetensors format."""
    print(f"\n{'='*80}")
    print(f"Saving quantized model to: {output_dir}")
    print(f"{'='*80}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights in safetensors format
    print("Saving model weights (safetensors format)...")
    state_dict = model.state_dict()
    
    # Convert state dict to CPU and ensure all tensors are contiguous
    state_dict_cpu = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    
    # Save as safetensors
    save_file(state_dict_cpu, output_path / "model.safetensors")
    
    # Create model index file for HF compatibility
    import json
    model_index = {
        "metadata": {"format": "pt"},
        "weight_map": {k: "model.safetensors" for k in state_dict_cpu.keys()}
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    # Add quantization config to HF config
    print("Saving config with quantization metadata...")
    quant_config = {
        "quant_method": "awq",
        "w_bits": args.w_bits,
        "group_size": args.group_size,
        "zero_point": args.zero_point,
        "version": "gemm",
        "calibration": {
            "dataset": args.calibration_dataset,
            "dataset_subset": args.calibration_subset,
            "n_samples": args.n_samples,
            "max_seq_len": args.max_seq_len,
            "apply_clip": args.apply_clip,
        }
    }
    
    # Add quantization_config to the HF config
    hf_config.quantization_config = quant_config
    
    # Save config (now includes quantization_config)
    hf_config.save_pretrained(output_path)
    
    print(f"\n✓ Model saved successfully to {output_dir}")
    print(f"  - Model weights: model.safetensors")
    print(f"  - Config with quantization metadata: config.json")
    print(f"  - Tokenizer files saved\n")
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        push_to_huggingface_hub(output_dir, args)


def push_to_huggingface_hub(output_dir: str, args):
    """Push the quantized model to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo
    
    # Determine the model ID
    if args.hub_model_id:
        repo_id = args.hub_model_id
    else:
        # Use the output directory basename as the model name
        repo_id = Path(output_dir).name
    
    print(f"\n{'='*80}")
    print(f"Pushing model to Hugging Face Hub: {repo_id}")
    print(f"{'='*80}\n")
    
    try:
        # Create repository (if it doesn't exist)
        print(f"Creating/checking repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            exist_ok=True,
            repo_type="model",
        )
        
        # Upload all files
        print(f"Uploading files from {output_dir}...")
        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload AWQ quantized model ({args.w_bits}-bit, group_size={args.group_size})",
        )
        
        print(f"\n✓ Model successfully pushed to Hugging Face Hub!")
        print(f"  Repository: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n✗ Failed to push to Hugging Face Hub: {e}")
        print("  Make sure you are logged in with `huggingface-cli login`\n")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("AWQ Quantization for Qwen3-0.6B")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Quantization: {args.w_bits}-bit, group_size={args.group_size}, zero_point={args.zero_point}")
    print(f"  Calibration: {args.calibration_dataset} ({args.n_samples} samples)")
    print(f"  Apply clipping: {args.apply_clip}")
    print(f"  Save model: {args.save_quantized_model}")
    if args.push_to_hub:
        hub_id = args.hub_model_id if args.hub_model_id else Path(args.output_dir).name
        print(f"  Push to Hub: Yes → {hub_id} (private={args.hub_private})")
    else:
        print(f"  Push to Hub: No")
    print(f"  Evaluate perplexity: {args.eval_ppl}\n")
    
    # Load model
    start_time = time.time()
    model, tokenizer, hf_config = load_qwen3_model(args.model_path)
    output_dtype = hf_config.torch_dtype
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f}s\n")
    
    # Optional: Evaluate perplexity before quantization
    # if args.eval_ppl:
    #     print("\n" + "="*80)
    #     print("Evaluating perplexity BEFORE quantization...")
    #     print("="*80 + "\n")
    #     try:
    #         from nanoquantization.benchmark.perplexity import calculate_perplexity
    #         ppl_before = calculate_perplexity(
    #             model=model,
    #             tokenizer=tokenizer,
    #             window_size=2048,
    #             stride=512,
    #             dataset_id="wikitext",
    #             dataset_subset="wikitext-103-v1",
    #             dataset_split="test",
    #         )
    #         print(f"\n✓ Perplexity BEFORE quantization: {ppl_before:.2f}\n")
    #     except Exception as e:
    #         print(f"Warning: Could not evaluate perplexity: {e}\n")
    
    # Initialize quantizer
    print("="*80)
    print("Starting AWQ quantization...")
    print("="*80 + "\n")
    
    quantizer = AWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        w_bits=args.w_bits,
        group_size=args.group_size,
        zero_point=args.zero_point,
        calibration_data=args.calibration_dataset,
        hf_dataset_subset=args.calibration_subset,
        hf_dataset_split="train",
        text_column="text",
        n_samples=args.n_samples,
        max_seq_len=args.max_seq_len,
        apply_clip=args.apply_clip,
        fake_quantization=False,
        output_dtype=output_dtype,
    )
    
    # Run quantization
    quant_start_time = time.time()
    quantizer.quantize()
    quant_time = time.time() - quant_start_time
    
    print("\n" + "="*80)
    print(f"✓ Quantization completed in {quant_time:.2f}s ({quant_time/60:.2f} minutes)")
    print("="*80 + "\n")
    
    # Move model back to CUDA after quantization (quantizer moves layers to CPU to save memory)
    # Also convert to float16 to match quantized layer dtype
    model.cuda()
    
    # Optional: Evaluate perplexity after quantization
    if args.eval_ppl:
        print("="*80)
        print("Evaluating perplexity AFTER quantization...")
        print("="*80 + "\n")
        try:
            from nanoquantization.benchmark.perplexity import calculate_perplexity
            ppl_after = calculate_perplexity(
                model=model,
                tokenizer=tokenizer,
                window_size=2048,
                stride=512,
                dataset_id="wikitext",
                dataset_subset="wikitext-103-v1",
                dataset_split="test",
            )
            print(f"\n✓ Perplexity AFTER quantization: {ppl_after:.2f}")
        except Exception as e:
            print(f"Warning: Could not evaluate perplexity: {e}\n")
    
    # Save quantized model (if enabled)
    if args.save_quantized_model:
        save_quantized_model(model, tokenizer, hf_config, args.output_dir, args)
    else:
        print("\n" + "="*80)
        print("Skipping model save (--save_quantized_model flag not set)")
        print("="*80 + "\n")
    
    total_time = time.time() - start_time
    print("="*80)
    print(f"✓ Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*80 + "\n")
    
    if args.save_quantized_model:
        print(f"Quantized model saved to: {args.output_dir}")
        print("\nOutput structure:")
        print("  ├── model.safetensors              (quantized model weights)")
        print("  ├── model.safetensors.index.json  (weight map)")
        print("  ├── config.json                    (model config + quantization metadata)")
        print("  ├── tokenizer.json                 (tokenizer)")
        print("  └── tokenizer_config.json          (tokenizer config)")
        print("\nThe model is saved in Hugging Face safetensors format for:")
        print("  • Faster loading times")
        print("  • Safe deserialization (no arbitrary code execution)")
        print("  • Better memory efficiency")
        print("  • Cross-platform compatibility")
        print("\nQuantization config is embedded in config.json under 'quantization_config' key")
    else:
        print("Quantization completed. Model not saved to disk.")


if __name__ == "__main__":
    main()

