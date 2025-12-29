"""
For some weights like qkv_proj in attention layer, we will pack multiple weights together for efficiency.
But in HF weights, they are separated tensors.
"""

import glob
import os
import torch
from torch import nn
from safetensors import safe_open


def default_weights_loader(params: nn.Parameter, weights: torch.Tensor) -> None:
    params.data.copy_(weights)


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
                param = model.get_parameter(weight_name)
                default_weights_loader(param, f.get_tensor(weight_name))
