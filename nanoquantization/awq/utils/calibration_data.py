from typing import Union, List
from transformers import AutoTokenizer
import torch
from datasets import load_dataset


def get_calibration_data(
    data: Union[str, List[str]] = "wikitext",
    tokenizer: AutoTokenizer = None,
    n_samples: int = 128,
    max_seq_len: int = 512,
    hf_dataset_subset: str = "wikitext-103-v1",
    hf_dataset_split: str = "train",
    text_column: str = "text",
) -> torch.Tensor:
    """
    Get calibration data
    """
    if isinstance(data, str):
        dataset = load_dataset(
            path=data, name=hf_dataset_subset, split=hf_dataset_split
        ).shuffle(seed=42)
    elif isinstance(data, List[str]):
        dataset = [{text_column: text} for text in data]
    else:
        raise ValueError(f"Invalid data type: {type(data)}")

    encodings = []
    n_runs = 0
    for data in dataset:
        line_encoded = tokenizer.encode(
            data[text_column].strip(), add_special_tokens=False
        )
        # add bos token if not present
        if (
            tokenizer
            and hasattr(tokenizer, "bos_token_id")
            and tokenizer.bos_token_id is not None
        ):
            # Check if it's already there (common in some pre-proc datasets)
            if len(line_encoded) > 0 and line_encoded[0] != tokenizer.bos_token_id:
                line_encoded = [tokenizer.bos_token_id] + line_encoded
        # skip if sequence length is less than max_seq_len
        if len(line_encoded) < max_seq_len:
            continue
        line_encoded = line_encoded[:max_seq_len]
        encodings.append(line_encoded)
        n_runs += 1
        if n_runs >= n_samples:
            break
    if len(encodings) < n_samples:
        print(
            f"Only found {len(encodings)} usable samples out of {n_samples} requested. Consider checking dataset length or reducing max_seq_len."
        )
    cali_data = torch.tensor(encodings, dtype=torch.long)
    print(f"Calibration data shape: {cali_data.shape}")
    return cali_data
