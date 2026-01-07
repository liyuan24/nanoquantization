from nanoquantization.models.qwen3 import Qwen3ForCausalLM
from nanoquantization.models.qwen3_quantized import Qwen3QuantizedForCausalLM
from nanoquantization.utils.loader import load_model
from transformers import AutoTokenizer, AutoConfig
import torch.nn as nn
from datasets import load_dataset
import torch
from tqdm import tqdm


def calculate_perplexity(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    window_size: int,
    stride: int,
    dataset_id: str,
    dataset_subset: str,
    dataset_split: str,
) -> float:
    """
    Calculate the perplexity of the model on the dataset.
    """
    dataset = load_dataset(dataset_id, dataset_subset, split=dataset_split)
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.shape[1]
    last_end_ind = 0
    cross_entropy_losses = []
    for i in tqdm(range(0, seq_len, stride), desc="Calculating perplexity"):
        end_ind = min(i + window_size, seq_len)
        input_ids = encodings.input_ids[:, i:end_ind]
        target_ids_in_window = end_ind - last_end_ind
        labels = input_ids.clone()
        # mask out the token ids that have been calcuated loss in last window
        labels[:, :-target_ids_in_window] = -100
        with torch.no_grad():
            cross_entropy_loss = model.compute_loss(input_ids[:, :-1], labels[:, 1:])
        cross_entropy_losses.append(cross_entropy_loss * target_ids_in_window)
        last_end_ind = end_ind
    ppl = torch.exp(torch.stack(cross_entropy_losses).sum() / seq_len)
    return ppl.item()


if __name__ == "__main__":
    weight_path = "/workspace/quantized_models/qwen3_0pt6b_awq"
    tokenizer = AutoTokenizer.from_pretrained(weight_path)
    hf_config = AutoConfig.from_pretrained(weight_path)
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")
    hf_config = AutoConfig.from_pretrained(weight_path)
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
    quantized_model.to("cuda")
    ppl = calculate_perplexity(
        quantized_model,
        tokenizer,
        window_size=2028,
        stride=512,
        dataset_id="wikitext",
        dataset_subset="wikitext-2-v1",
        dataset_split="test",
    )
    print(f"Perplexity for Qwen3-0.6B quantized model: {ppl:.2f}")
