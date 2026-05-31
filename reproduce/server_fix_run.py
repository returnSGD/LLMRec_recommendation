"""Fix embed_tokens on server and restart training — v2"""
import paramiko
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HOST = "connect.westd.seetacloud.com"
PORT = 37242
USER = "root"
PASS = "eKKtLzujQdlg"

def run(cmd, timeout=30):
    print(f"\n>>> {cmd[:150]}{'...' if len(cmd) > 150 else ''}")
    chan = ssh.get_transport().open_session()
    chan.set_combine_stderr(False)
    chan.exec_command(cmd)
    out = chan.recv(65536).decode("utf-8", errors="replace")
    err = chan.recv_stderr(65536).decode("utf-8", errors="replace")
    while not chan.exit_status_ready():
        time.sleep(0.2)
        try:
            more = chan.recv(65536).decode("utf-8", errors="replace")
            if more:
                out += more
        except:
            pass
    code = chan.recv_exit_status()
    if out:
        print(out[-2000:])
    if err:
        print(f"STDERR({code}): {err[-500:]}")
    return out, err, code

print("Connecting...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print("Connected!")

# Step 0: Kill existing training first
print("\n=== Step 0: Kill existing training ===")
run("pkill -f 'train.py' 2>/dev/null || true; sleep 1")
out, _, _ = run("ps aux | grep '[t]rain.py' | wc -l")
if "0" in out:
    print("[OK] Training stopped")

# Step 1: Write fix script to server file (avoids shell escaping issues)
print("\n=== Step 1: Write fix script to server ===")
fix_script = '''
with open("/root/LLM-Rec/reproduce/modeling_p5.py", "r") as f:
    content = f.read()

# Replace encoder line
old = "self.encoder = JointEncoder(encoder_config)\\n"
new = "self.encoder = JointEncoder(encoder_config)\\n        self.encoder.embed_tokens = self.shared\\n"
content = content.replace(old, new)

# Replace decoder line
old = "self.decoder = T5Stack(decoder_config)\\n"
new = "self.decoder = T5Stack(decoder_config)\\n        self.decoder.embed_tokens = self.shared\\n"
content = content.replace(old, new)

with open("/root/LLM-Rec/reproduce/modeling_p5.py", "w") as f:
    f.write(content)
print("Fix applied")
'''

# Write via SFTP
sftp = ssh.open_sftp()
with sftp.open("/tmp/fix_embed.py", "w") as f:
    f.write(fix_script)
sftp.close()
print("[OK] Script written to /tmp/fix_embed.py")

# Step 2: Execute fix
print("\n=== Step 2: Execute fix ===")
run("/root/miniconda3/bin/python /tmp/fix_embed.py")

# Step 3: Verify
print("\n=== Step 3: Verify fix ===")
out, _, _ = run("sed -n '191,205p' /root/LLM-Rec/reproduce/modeling_p5.py")
if out.count("embed_tokens = self.shared") >= 2:
    print("[OK] Both embed_tokens assignments added!")
else:
    print(f"[FAIL] Expected 2 embed_tokens, found {out.count('embed_tokens = self.shared')}")
    # Try alternative: exact line-based approach
    print("Trying sed approach instead...")
    run("sed -i '193 a\\        self.encoder.embed_tokens = self.shared' /root/LLM-Rec/reproduce/modeling_p5.py")
    run("sed -i '201 a\\        self.decoder.embed_tokens = self.shared' /root/LLM-Rec/reproduce/modeling_p5.py")
    out, _, _ = run("sed -n '191,210p' /root/LLM-Rec/reproduce/modeling_p5.py")
    if out.count("embed_tokens = self.shared") >= 2:
        print("[OK] Sed approach worked!")

# Step 4: Syntax check
print("\n=== Step 4: Syntax check ===")
out, err, code = run("cd /root/LLM-Rec && /root/miniconda3/bin/python -c \"import ast; ast.parse(open('reproduce/modeling_p5.py').read()); print('Syntax OK')\"")
if code != 0:
    print("[FAIL] Syntax error, aborting!")
    ssh.close()
    sys.exit(1)

# Step 5: Full model smoke test (need losses on config)
print("\n=== Step 5: Full model loading test ===")
smoke_cmd = """cd /root/LLM-Rec && HF_HUB_OFFLINE=1 /root/miniconda3/bin/python -c "
import sys; sys.path.insert(0,'reproduce')
from transformers import T5Config
config = T5Config.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small')
config.losses = 'sequential,explanation,review,traditional'
from train import P5Pretraining
model = P5Pretraining.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small', config=config)
print('Model loaded OK, params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
"
"""
out, err, code = run(smoke_cmd, timeout=120)
if "Model loaded OK" in out:
    print("[OK] Model loading test PASSED!")
else:
    print("[FAIL] Model loading test FAILED")
    print(f"stderr: {err[-500:]}")
    ssh.close()
    sys.exit(1)

# Step 6: Launch training
print("\n=== Step 6: Launch training ===")
run("cd /root/LLM-Rec && > train.log 2>/dev/null; HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 nohup /root/miniconda3/bin/python -u reproduce/train.py --backbone /root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small --dataset beauty --epochs 10 --batch_size 16 --lr 1e-3 --output_dir ./output </dev/null >train.log 2>&1 &")

time.sleep(8)

# Step 7: Check running
print("\n=== Step 7: Verify training ===")
out, _, _ = run("ps aux | grep '[t]rain.py'")
if "train.py" in out:
    print("[OK] Training process running!")
else:
    print("[WARN] No training process")

# Step 8: Log + GPU
print("\n=== Step 8: Training log ===")
time.sleep(3)
out, _, _ = run("tail -50 /root/LLM-Rec/train.log")
print("--- End of log ---")

print("\n=== Step 9: GPU status ===")
out, _, _ = run("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader")
print(f"GPU: {out.strip()}")

# Check if training has passed model loading
out2, _, _ = run("grep -c 'Epoch' /root/LLM-Rec/train.log 2>/dev/null || echo 0")
if out2.strip() != "0":
    print(f"[OK] Training is in epoch phase ({out2.strip()} epoch lines found)")

ssh.close()
print("\nDone!")
