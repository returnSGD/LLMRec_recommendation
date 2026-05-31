#!/usr/bin/env python3
"""SSH helper to run commands on remote server."""
import subprocess
import sys
import os

HOST = "connect.westd.seetacloud.com"
PORT = "37242"
USER = "root"
PASS = "eKKtLzujQdlg"

def ssh_run(command, timeout=120):
    """Run a command on the remote server via SSH."""
    # Use SSH_ASKPASS trick to pass password non-interactively
    env = os.environ.copy()
    env["SSH_ASKPASS"] = sys.executable  # dummy, we'll use the pass via a temp script

    import tempfile
    # Write password to a temp script
    fd, askpass_script = tempfile.mkstemp(suffix='.py', prefix='ssh_askpass_')
    with os.fdopen(fd, "w") as f:
        f.write(f'import sys\nsys.stdout.write("{PASS}\\n")\nsys.stdout.flush()\n')
    env["SSH_ASKPASS"] = askpass_script
    env["DISPLAY"] = "dummy:0"

    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", PORT,
        f"{USER}@{HOST}",
        command
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    os.unlink(askpass_script)

    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def ssh_run_batch(commands, timeout=120):
    """Run multiple commands in one SSH session."""
    script = " && ".join(commands)
    return ssh_run(script, timeout)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ssh_helper.py '<command>'")
        sys.exit(1)

    cmd = sys.argv[1]
    stdout, stderr, rc = ssh_run(cmd)
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    sys.exit(rc)
