from nanoquantization.sampling_params import SamplingParams
from transformers import AutoTokenizer
from nanoquantization.engine.llm_engine import LLMEngine


def main():
    model_path = "/workspace/huggingface/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = ["write me a haiku about AI", "tell me a joke about AI"]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [
        tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    llm_engine = LLMEngine(model_path)
    outputs = llm_engine.generate(texts, sampling_params)
    for text, output in zip(texts, outputs):
        print("=" * 100)
        print(f"Prompt: \n{text!r}")
        print(f"Output: \n{output!r}")


if __name__ == "__main__":
    main()
