"""Upload ablation experiment to server."""
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
        print(out.strip()[-500:])
    if err.strip():
        print("[stderr]", err.strip()[-300:])
    return out, err

# 1. Upload ablation.py
print("Uploading src/ablation.py...")
sftp.put('src/ablation.py', f'{SERVER_DIR}/src/ablation.py')
print("   Done.")

# 2. Create run_ablation.sh
print("\nCreating run_ablation.sh...")
script = f'''#!/bin/bash
set -e
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
cd {SERVER_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="{SERVER_DIR}/logs"
mkdir -p $LOG_DIR

BEST_CKPT=$(ls -t output/*/Epoch10.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t output/*/BEST_EVAL_LOSS.pth 2>/dev/null | head -1)
fi
echo "Using checkpoint: $BEST_CKPT"

exec > >(tee -a "$LOG_DIR/ablation_$TIMESTAMP.log") 2>&1

echo "============================================"
echo "Ablation Experiment Started: $(date)"
echo "Checkpoint: $BEST_CKPT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python -m src.ablation \
    --dataset beauty \
    --backbone t5-small \
    --p5_checkpoint "$BEST_CKPT" \
    --sample_ratio 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --max_eval 500 \
    --beam_size 20 \
    --output_dir ./outputs/ablation

echo ""
echo "============================================"
echo "Ablation Complete: $(date)"
echo "============================================"
'''

sftp.putfo(io.BytesIO(script.encode()), f"{SERVER_DIR}/run_ablation.sh")
run(f"chmod +x {SERVER_DIR}/run_ablation.sh", "chmod")

# 3. Verify
print("\n=== Verification ===")
out, _ = run(f"head -20 {SERVER_DIR}/run_ablation.sh")
print(out)

print()
out, _ = run(f"ls -la {SERVER_DIR}/src/ablation.py")
print(out)

# 4. Check current server status
print("\n=== Server Status ===")
out, _ = run("ps aux | grep 'eval.py\\|rl_experiment' | grep -v grep | head -2")
print(out or "No training/eval process")

out, _ = run("grep '^---' /root/LLM-Rec/logs/train_20260531_124143.log | tail -5")
print("\nEval tasks:")
print(out)

sftp.close()
client.close()

print("\nDone. Run on server: nohup bash /root/LLM-Rec/run_ablation.sh &")
