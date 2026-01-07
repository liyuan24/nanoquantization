"""
For some weights like qkv_proj in attention layer, we will pack multiple weights together for efficiency.
But in HF weights, they are separated tensors.
"""

import glob
import os
from nanoquantization.models.qwen3_quantized import Qwen3QuantizedForCausalLM
import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig


def default_weights_loader(tensor: torch.Tensor, weights: torch.Tensor) -> None:
    """Load weights into a tensor (can be a parameter or buffer)."""
    tensor.data.copy_(weights)


def load_model(model: nn.Module, path: str) -> None:
    """
    A simple weight loader

    Arguments:
        model: the model to load weights into
        path: the path to the checkpoint
    Returns:
        None
    """
    # use all tensor files in the checkpoint directory
    tensor_files = glob.glob(os.path.join(path, "*.safetensors"))
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as f:
            # example weight_name: model.layers.0.self_attn.q_proj.weight
            for weight_name in f.keys():
                print(f"weight_name: {weight_name}")
                # Try to get as parameter first, then as buffer
                try:
                    tensor = model.get_parameter(weight_name)
                except AttributeError:
                    # If not a parameter, try to get as buffer (for quantized layers like qscales, qzeros, qweight)
                    try:
                        tensor = model.get_buffer(weight_name)
                    except AttributeError:
                        raise AttributeError(
                            f"`{weight_name}` is neither a parameter nor a buffer in the model"
                        )
                default_weights_loader(tensor, f.get_tensor(weight_name))


if __name__ == "__main__":
    weight_path = "/workspace/quantized_models/qwen3_0pt6b_awq"
    hf_config = AutoConfig.from_pretrained(weight_path)
    print(hf_config)
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
