#!/bin/bash
# Fix train.py syntax and start training
cd /root/LLM-Rec

# Fix the backbone line (add missing closing paren)
sed -i "117s/'t5-small'$/'t5-small')/" reproduce/train.py

# Verify
/root/miniconda3/bin/python -c "
import ast
ast.parse(open('reproduce/train.py').read())
print('Syntax OK!')
"

if [ $? -eq 0 ]; then
    echo "Starting training..."
    HF_HUB_OFFLINE=1 nohup /root/miniconda3/bin/python reproduce/train.py \
        --backbone /root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small \
        --dataset beauty \
        --epochs 10 \
        --batch_size 16 \
        --lr 1e-3 \
        --output_dir ./output \
        > train.log 2>&1 &
    echo "PID: $!"
    sleep 5
    head -20 train.log
fi
