#!/bin/bash
# One-click setup for LLM fine-tuning with Unsloth on a rented GPU (Vast.ai and similar).
# Installs Unsloth and downloads the notebook + example data into /workspace.
set -e

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo "==> Downloading the fine-tuning kit"
if [ ! -d unsloth-finetune-cloud ]; then
  git clone --depth 1 https://github.com/giovannajasmin/unsloth-finetune-cloud.git
fi

echo "==> Installing Unsloth (a few minutes the first time)"
python3 -m pip install --upgrade pip
python3 -m pip install unsloth

echo "==> Ready."
echo "Open Jupyter and go to $WORKDIR/unsloth-finetune-cloud/finetune.ipynb"
echo "or run: cd $WORKDIR/unsloth-finetune-cloud && python3 train.py"
