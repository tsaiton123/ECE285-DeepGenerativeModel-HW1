#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fL -C - --retry 5 --retry-all-errors \
  'https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia' \
  -o data/chest-xray-pneumonia.zip
python -m chest_vae.data
