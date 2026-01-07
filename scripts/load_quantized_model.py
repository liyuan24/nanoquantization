import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
from nanoquantization.models.qwen3_quantized import Qwen3QuantizedForCausalLM
from nanoquantization.utils.loader import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a quantized Qwen3 model")
    parser.add_argument(
        "--local_path",
        type=str,
        default="/workspace/quantized_models/qwen3_0pt6b_awq",
        help="Path to the local quantized model directory",
    )
    args = parser.parse_args()

    weight_path = args.local_path
    tokenizer = AutoTokenizer.from_pretrained(weight_path)
    hf_config = AutoConfig.from_pretrained(weight_path)
    torch.set_default_dtype(hf_config.torch_dtype)
    quantized_model = Qwen3QuantizedForCausalLM(
        num_hidden_layers=hf_config.num_hidden_layers,
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        total_num_heads=hf_config.num_attention_heads,
        total_num_kv_heads=hf_config.num_key_value_heads,
        max_position_embeddings=hf_config.max_position_embeddings,
        intermediate_size=hf_config.intermediate_size,
        w_bits=hf_config.quantization_config["w_bits"],
        group_size=hf_config.quantization_config["group_size"],
        head_dim=hf_config.head_dim,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
    )
    load_model(quantized_model, weight_path)
