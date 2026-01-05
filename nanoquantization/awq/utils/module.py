from typing import Dict
import torch.nn as nn


def get_named_linear(module: nn.Module) -> Dict[str, nn.Linear]:
    named_linear = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            named_linear[name] = submodule
    return named_linear


def get_op_name(transformer_block: nn.Module, op: nn.Module) -> str:
    for name, submodule in transformer_block.named_modules():
        if submodule is op:
            return name
    raise ValueError(f"op {op} not found in transformer_block {transformer_block}")


def get_op_by_name(module: nn.Module, name: str) -> nn.Module:
    """
    Get the submodule by its relative name in the module.
    """
    for module_name, submodule in module.named_modules():
        if module_name == name:
            return submodule
    raise ValueError(f"op {name} not found in module {module}")

# TransformerBlock (layer)
# ├── self_attn (Module)
# │   ├── q_proj (Linear)
# │   ├── k_proj (Linear)
# │   └── v_proj (Linear)
# └── mlp (Module)
#     ├── gate_proj (Linear)
#     └── up_proj (Linear)
def set_op_by_name(module: nn.Module, name: str, op: nn.Module) -> None:
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for level in levels[:-1]:
            if level.isdigit():
                mod_ = mod_[int(level)]
            else:
                mod_ = getattr(mod_, level)
        setattr(mod_, levels[-1], op)
    else:
        setattr(module, name, op)