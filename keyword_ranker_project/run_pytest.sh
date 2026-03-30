#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -v
