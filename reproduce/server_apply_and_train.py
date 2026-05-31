"""Apply keys_to_ignore fix, test, and launch training"""
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
    print(f"\n>>> {cmd[:180]}{'...' if len(cmd) > 180 else ''}")
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

# Step 1: Upload simple fix script using SFTP
print("\n=== Step 1: Upload fix script ===")

# Simpler fix: just change [ to { at the right lines, and the closing ] to }
fix_script = """import re

path = '/root/LLM-Rec/reproduce/modeling_p5.py'
with open(path, 'r') as f:
    content = f.read()

# Replace opening brackets for these specific attributes
content = content.replace(
    '_keys_to_ignore_on_load_missing = [',
    '_keys_to_ignore_on_load_missing = {'
)
content = content.replace(
    '_keys_to_ignore_on_load_unexpected = [',
    '_keys_to_ignore_on_load_unexpected = {'
)

# Replace the closing brackets - find the pattern:
# For load_missing: the ] that comes before _keys_to_ignore_on_load_unexpected
old1 = '''    ]
    _keys_to_ignore_on_load_unexpected'''
new1 = '''    }
    _keys_to_ignore_on_load_unexpected'''
content = content.replace(old1, new1)

# For load_unexpected: the ] that comes before def __init__
old2 = '''    ]

    def __init__(self, config):'''
new2 = '''    }

    def __init__(self, config):'''
content = content.replace(old2, new2)

with open(path, 'w') as f:
    f.write(content)

# Verify
with open(path, 'r') as f:
    result = f.read()

if '_keys_to_ignore_on_load_missing = {' in result and '_keys_to_ignore_on_load_unexpected = {' in result:
    print('OK: lists converted to sets')
else:
    print('WARN: pattern not matched')
    # Show relevant lines
    for i, line in enumerate(result.split('\\n'), 1):
        if '_keys_to_ignore' in line:
            print(f'  L{i}: {line}')
"""

sftp = ssh.open_sftp()
with sftp.open("/tmp/fix_keys.py", "w") as f:
    f.write(fix_script)
sftp.close()
print("[OK] Uploaded")

# Step 2: Apply
print("\n=== Step 2: Apply fix ===")
run("/root/miniconda3/bin/python /tmp/fix_keys.py")

# Step 3: Verify fix
print("\n=== Step 3: Verify ===")
out, _, _ = run("grep -A5 '_keys_to_ignore_on_load_missing' /root/LLM-Rec/reproduce/modeling_p5.py | head -8")
if "{ " in out:
    print("[OK] Set literals found")

# Step 4: Syntax check
out, err, code = run("cd /root/LLM-Rec && /root/miniconda3/bin/python -c \"import ast; ast.parse(open('reproduce/modeling_p5.py').read()); print('Syntax OK')\"")
if code != 0:
    print("[FAIL] Syntax error!")
    ssh.close()
    sys.exit(1)

# Step 5: Model test
print("\n=== Step 5: Model loading test ===")
test_cmd = """cd /root/LLM-Rec && HF_HUB_OFFLINE=1 /root/miniconda3/bin/python -c "
import sys; sys.path.insert(0,'reproduce')
from transformers import T5Config
config = T5Config.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small')
config.losses = 'sequential,explanation,review,traditional'
from train import P5Pretraining
model = P5Pretraining.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small', config=config)
print('Model loaded OK, params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
"
"""
out, err, code = run(test_cmd, timeout=120)
if "Model loaded OK" in out:
    print("[OK] Model loading test PASSED!")
else:
    print("[FAIL] Model test failed")
    ssh.close()
    sys.exit(1)

# Step 6: Launch training
print("\n=== Step 6: Launch training ===")
run("cd /root/LLM-Rec && > train.log 2>/dev/null; HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 nohup /root/miniconda3/bin/python -u reproduce/train.py --backbone /root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small --dataset beauty --epochs 10 --batch_size 16 --lr 1e-3 --output_dir ./output </dev/null >train.log 2>&1 &")

time.sleep(10)

# Step 7: Check
print("\n=== Step 7: Training status ===")
out, _, _ = run("ps aux | grep '[t]rain.py'")
if "train.py" in out:
    print("[OK] Training process running")
else:
    print("[WARN] No training process found")

# Step 8: Log
print("\n=== Step 8: Training log ===")
out, _, _ = run("tail -60 /root/LLM-Rec/train.log")
print("--- End ---")

# Step 9: GPU
print("\n=== Step 9: GPU ===")
out, _, _ = run("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader")
print(f"GPU: {out.strip()}")

# Step 10: Wait 20s and recheck
print("\n=== Step 10: Wait 20s for data loading to complete ===")
time.sleep(20)
out, _, _ = run("tail -40 /root/LLM-Rec/train.log")
print("--- Updated log ---")
out2, _, _ = run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
print(f"GPU: {out2.strip()}")

# Check for errors in log
out3, _, _ = run("grep -i 'error\|traceback\|exception' /root/LLM-Rec/train.log 2>/dev/null | tail -10")
if out3.strip():
    print(f"\n[WARN] Errors in log:\n{out3}")

ssh.close()
print("\nDone!")
