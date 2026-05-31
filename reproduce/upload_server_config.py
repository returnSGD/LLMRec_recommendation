"""Upload updated rl_experiment.py and run_train.sh with server max config."""
import paramiko
import io

SERVER_HOST = "connect.westd.seetacloud.com"
SERVER_PORT = 37242
SERVER_USER = "root"
SERVER_PASS = "eKKtLzujQdlg"
SERVER_DIR = "/root/LLM-Rec"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER,
               password=SERVER_PASS, timeout=30)
sftp = client.open_sftp()

def run(cmd, desc=""):
    if desc:
        print(f"[{desc}]")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        print(out.strip()[-300:])
    if err.strip():
        print("[stderr]", err.strip()[-200:])
    return out, err

# 1. Upload rl_experiment.py
print("1. Uploading rl_experiment.py...")
sftp.put('src/rl_experiment.py', f'{SERVER_DIR}/src/rl_experiment.py')
print("   Done.")

# 2. Check current eval progress
print("\n2. Current server status:")
out, _ = run("ps aux | grep 'eval.py' | grep -v grep | head -1")
print(f"   Eval: {'Running' if out.strip() else 'DONE'}")
out, _ = run("grep -c '^---' /root/LLM-Rec/logs/train_20260531_124143.log")
print(f"   Eval tasks done: {out.strip()}")

# 3. Update run_train.sh with server max config
print("\n3. Updating run_train.sh with max config...")
script = f'''#!/bin/bash
set -e
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
cd {SERVER_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="{SERVER_DIR}/logs"
mkdir -p $LOG_DIR

exec > >(tee -a "$LOG_DIR/train_$TIMESTAMP.log") 2>&1

echo "============================================"
echo "LLM-Rec Training Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "============================================"

# --- Phase 1: P5 Baseline Training ---
echo ""
echo "##############################################"
echo "# Phase 1: P5 Baseline Training (10 epochs)"
echo "##############################################"
echo ""

python reproduce/train.py \
    --dataset beauty \
    --backbone t5-small \
    --epochs 10 \
    --batch_size 32 \
    --sample_ratio 0.05 \
    --output_dir ./output \
    --fp16 \
    --lr 1e-3 \
    --warmup_ratio 0.05 \
    --seed 2022

echo ""
echo "P5 Training Complete: $(date)"

BEST_CKPT=$(ls -t output/*/Epoch10.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t output/*/BEST_EVAL_LOSS.pth 2>/dev/null | head -1)
fi
echo "Best checkpoint: $BEST_CKPT"

# --- Phase 2: P5 Evaluation ---
echo ""
echo "##############################################"
echo "# Phase 2: P5 Baseline Evaluation"
echo "##############################################"
echo ""

if [ -n "$BEST_CKPT" ]; then
    python reproduce/eval.py \
        --checkpoint "$BEST_CKPT" \
        --dataset beauty \
        --backbone t5-small \
        --batch_size 8 \
        --beam_size 20 \
        --output_file "$LOG_DIR/p5_eval_$TIMESTAMP.json"
fi

# --- Phase 3: RL + Memory (96GB max config) ---
echo ""
echo "##############################################"
echo "# Phase 3: RL + Memory (96GB max config)"
echo "#   memory_dim=512, hidden_dim=512, batch=512"
echo "#   n_layers=4, top_k=50, all items"
echo "##############################################"
echo ""

if [ -n "$BEST_CKPT" ]; then
    python -m src.rl_experiment \
        --dataset beauty \
        --backbone t5-small \
        --p5_checkpoint "$BEST_CKPT" \
        --data_dir data \
        --config_preset server \
        --sample_ratio 0.05 \
        --epochs 10 \
        --batch_size 512 \
        --beam_size 20 \
        --max_eval 5000 \
        --output_dir ./outputs/rl_memory
fi

echo ""
echo "============================================"
echo "ALL TRAINING COMPLETE: $(date)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
echo "============================================"
'''

sftp.putfo(io.BytesIO(script.encode()), f"{SERVER_DIR}/run_train.sh")
run(f"chmod +x {SERVER_DIR}/run_train.sh", "chmod")

# Verify
print("\n4. Verification:")
out, _ = run(f"grep 'config_preset\\|batch_size.*512\\|hidden_dim=512\\|top_k=50' {SERVER_DIR}/run_train.sh")
print(out)

sftp.close()
client.close()
print("\nDone! Server config maxed for 96GB GPU.")
