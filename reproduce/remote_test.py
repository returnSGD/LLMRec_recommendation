import sys
sys.path.insert(0, "reproduce")

from transformers import T5Config
MODEL_PATH = "/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small"

config = T5Config.from_pretrained(MODEL_PATH)
print("Config OK, d_model:", config.d_model)

from tokenization import P5Tokenizer
tok = P5Tokenizer.from_pretrained(MODEL_PATH, max_length=512, do_lower_case=True)
print("Tokenizer OK, vocab:", tok.vocab_size)

from modeling_p5 import P5
from train import P5Pretraining
model = P5Pretraining.from_pretrained(MODEL_PATH, config=config)
n = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model OK, params: {n:.2f}M")
print("ALL TESTS PASSED")
