"""Fix server setup — use correct miniconda python path and install deps."""
import paramiko
import os
import io
from pathlib import Path

SERVER_HOST = "connect.westd.seetacloud.com"
SERVER_PORT = 37242
SERVER_USER = "root"
SERVER_PASS = "eKKtLzujQdlg"
SERVER_DIR = "/root/LLM-Rec"

PY = "/root/miniconda3/bin/python"
PIP = "/root/miniconda3/bin/pip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER,
               password=SERVER_PASS, timeout=30)
sftp = client.open_sftp()

def run(cmd, desc=""):
    if desc:
        print(f"\n[{desc}]")
    print(f"$ {cmd[:150]}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        print(out.strip()[-1000:])
    if err.strip():
        print("[stderr]", err.strip()[-500:])
    return out, err

# 1. Verify environment
print("=" * 60)
print("Verifying server environment...")
print("=" * 60)
run(f"{PY} --version", "Python version")
run(f"{PY} -c 'import torch; print(\"torch\", torch.__version__)'", "PyTorch")
run(f"{PY} -c 'import transformers; print(\"transformers\", transformers.__version__)'", "Transformers")
run(f"ls /root/miniconda3/bin/python*", "Python paths")

# 2. Install missing deps
print("\n" + "=" * 60)
print("Installing missing dependencies...")
print("=" * 60)
run(f"{PIP} install faiss-cpu 2>&1 | tail -5", "faiss-cpu")

# 3. Verify installs
print("\n" + "=" * 60)
print("Verifying installations...")
print("=" * 60)
run(f"{PY} -c 'import sentencepiece; print(\"spm OK\")'", "verify spm")
run(f"{PY} -c 'import faiss; print(\"faiss OK\")'", "verify faiss")
run(f"{PY} -c 'import nltk; print(\"nltk OK\")'", "verify nltk")
run(f"{PY} -c 'import rouge_score; print(\"rouge OK\")'", "verify rouge")

# 4. Create correct run_train.sh with conda python path
print("\n" + "=" * 60)
print("Creating training script with correct Python path...")
print("=" * 60)
train_script = f'''#!/bin/bash
# LLM-Rec Training — P5 Baseline + RL Memory
# GPU: RTX PRO 6000 (96GB), 22 vCPU, 110GB RAM
# Config: 5% data, 10 epochs, batch=32 (paper protocol)

set -e
export PATH="/root/miniconda3/bin:$PATH"
cd {SERVER_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="{SERVER_DIR}/logs"
mkdir -p $LOG_DIR

# Redirect all output to log file + terminal
exec > >(tee -a "$LOG_DIR/train_$TIMESTAMP.log") 2>&1

echo "============================================"
echo "LLM-Rec Training Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "FAISS: $(python -c 'import faiss; print(faiss.__version__)')"
echo "Log: $LOG_DIR/train_$TIMESTAMP.log"
echo "============================================"

# ── Phase 1: P5 Baseline Training ──
echo ""
echo "##############################################"
echo "# Phase 1: P5 Baseline Training (10 epochs)"
echo "##############################################"
echo ""

python reproduce/train.py \\
    --dataset beauty \\
    --backbone t5-small \\
    --epochs 10 \\
    --batch_size 32 \\
    --sample_ratio 0.05 \\
    --output_dir ./output \\
    --fp16 \\
    --lr 1e-3 \\
    --warmup_ratio 0.05 \\
    --seed 2022

echo ""
echo "============================================"
echo "P5 Training Complete: $(date)"
echo "============================================"

# Find the best checkpoint
BEST_CKPT=$(ls -t output/*/BEST_EVAL_LOSS.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t output/*/Epoch10.pth 2>/dev/null | head -1)
fi
echo "Best checkpoint: $BEST_CKPT"

# ── Phase 2: P5 Evaluation ──
echo ""
echo "##############################################"
echo "# Phase 2: P5 Baseline Evaluation"
echo "##############################################"
echo ""

if [ -n "$BEST_CKPT" ]; then
    python reproduce/eval.py \\
        --checkpoint "$BEST_CKPT" \\
        --dataset beauty \\
        --backbone t5-small \\
        --batch_size 8 \\
        --beam_size 20 \\
        --output_file "$LOG_DIR/p5_eval_$TIMESTAMP.json"
fi

# ── Phase 3: RL + Memory Experiment ──
echo ""
echo "##############################################"
echo "# Phase 3: RL + Memory Comparison Experiment"
echo "##############################################"
echo ""

if [ -n "$BEST_CKPT" ]; then
    python -m src.rl_experiment \\
        --dataset beauty \\
        --backbone t5-small \\
        --p5_checkpoint "$BEST_CKPT" \\
        --data_dir data \\
        --sample_ratio 0.05 \\
        --epochs 10 \\
        --batch_size 16 \\
        --beam_size 20 \\
        --max_eval 5000 \\
        --output_dir ./outputs/rl_memory
fi

echo ""
echo "============================================"
echo "ALL TRAINING COMPLETE: $(date)"
echo "GPU Summary:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
echo "============================================"
'''

sftp.putfo(io.BytesIO(train_script.encode()), f"{SERVER_DIR}/run_train.sh")
run(f"chmod +x {SERVER_DIR}/run_train.sh", "make executable")
print("   Script updated (using /root/miniconda3/bin/python).")

# 5. Quick smoke test
print("\n" + "=" * 60)
print("Running quick smoke test...")
print("=" * 60)
run(f"cd {SERVER_DIR} && {PY} -c \"from reproduce.tokenization import P5Tokenizer; t = P5Tokenizer.from_pretrained('t5-small', max_length=512, do_lower_case=True); print('Tokenizer OK, vocab:', t.vocab_size)\"", "tokenizer test")
run(f"cd {SERVER_DIR} && {PY} -c \"from reproduce.modeling_p5 import P5; from transformers import T5Config; c = T5Config.from_pretrained('t5-small'); c.losses='rating,sequential'; m = P5.from_pretrained('t5-small', config=c); print('Model loaded, params:', sum(p.numel() for p in m.parameters())/1e6, 'M')\"", "model test")
run(f"cd {SERVER_DIR} && {PY} -c \"from src.memory import MemoryManager; m = MemoryManager(memory_dim=128); print('Memory OK')\"", "memory test")
run(f"cd {SERVER_DIR} && {PY} -c \"from src.rl.actions import NUM_ACTIONS; print('RL actions OK, count:', NUM_ACTIONS)\"", "rl actions test")

# 6. Summary
print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print(f"""
Server: {SERVER_HOST}:{SERVER_PORT}
Project: {SERVER_DIR}
Python: {PY}
Training script: {SERVER_DIR}/run_train.sh

Start training (background):
  ssh -p {SERVER_PORT} root@{SERVER_HOST}
  cd {SERVER_DIR}
  nohup bash run_train.sh &

Monitor:
  tail -f {SERVER_DIR}/logs/train_*.log
""")

sftp.close()
client.close()
