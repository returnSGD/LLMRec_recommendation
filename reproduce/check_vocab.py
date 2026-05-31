import json, sys
sys.path.insert(0,'/root/LLM-Rec/reproduce')
with open('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small/config.json') as f:
    c = json.load(f)
    print('config vocab_size:', c.get('vocab_size'))

from tokenization import P5Tokenizer
tok = P5Tokenizer.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small', max_length=512)
print('tokenizer vocab_size:', tok.vocab_size)

from transformers import T5Config
cfg = T5Config.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small')
print('T5Config vocab_size:', cfg.vocab_size)
