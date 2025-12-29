from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

# Replace this with your actual model ID (e.g., path to local folder or HF Hub ID)
model_id = "Qwen/Qwen3-0.6B"

# 1. Load the configuration
# 'trust_remote_code=True' is often needed for newer architectures
# if they define custom code not yet in the official transformers library.
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print("=" * 100)
print(config)
print(config.architectures)
print("=" * 100)

# 2. Initialize the skeleton model (Zero RAM usage)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 3. Print the architecture
print(model)
