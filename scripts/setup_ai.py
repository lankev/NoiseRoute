"""Install the IBM local baseline in an isolated environment (not the training env)."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = root / ".venv-ai"
subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pip(*args):
    subprocess.run([str(python), "-m", "pip", "install", *args], check=True)


pip("torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
pip("qiskit==2.3.1", "qiskit-ibm-transpiler==0.18.0", "networkx==2.8.5", "numpy>=2,<3", "gymnasium", "PyYAML")
pip("--no-deps", "-e", str(root))
pip("--force-reinstall", "--no-deps", "networkx==2.8.5")
subprocess.run([str(python), "-m", "pip", "check"], check=True)
print(f"Worker ready: {python}")
print(f'PowerShell: $env:NOISEROUTE_AI_PYTHON = "{python}"')
print(f'Bash: export NOISEROUTE_AI_PYTHON="{python}"')
