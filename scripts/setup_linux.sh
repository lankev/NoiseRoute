#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ "${1:-cpu}" == cuda ]]; then
  .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
else
  .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -e '.[train,bench,export,demo,dev]'
if command -v cargo >/dev/null; then .venv/bin/python -m pip install ./rust; fi
.venv/bin/python -m noiseroute doctor --out runs/doctor.json
