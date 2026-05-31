#!/bin/bash
# Run on server to test and launch training
set -e

cd ~/LLM-Rec
MODEL_PATH=/root/pretrained_models/AI-ModelScope/t5-small

echo "=== Step 1: Test local model loading ==="
HF_HUB_OFFLINE=1 python -c "
import sys; sys.path.insert(0, 'reproduce')
from transformers import T5Config
config = T5Config.from_pretrained('$MODEL_PATH')
print('Config OK, d_model:', config.d_model)
from tokenization import P5Tokenizer
tok = P5Tokenizer.from_pretrained('$MODEL_PATH', max_length=512, do_lower_case=True)
print('Tokenizer OK, vocab:', tok.vocab_size)
from modeling_p5 import P5
from train import P5Pretraining
model = P5Pretraining.from_pretrained('$MODEL_PATH', config=config)
print('Model loaded OK, params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
print('ALL LOCAL MODEL LOADING TESTS PASSED')
"

echo ""
echo "=== Step 2: Launch training ==="
HF_HUB_OFFLINE=1 python reproduce/train.py \
    --backbone "$MODEL_PATH" \
    --dataset beauty \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-3 \
    --output_dir ./output
