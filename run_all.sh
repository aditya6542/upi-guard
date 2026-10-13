#!/usr/bin/env bash
# One-command reproduction. Order matters: generate -> train -> mutate -> demo.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/3  generating dataset"
python3 src/generator/generate.py

echo "==> 2/3  training + metrics"
python3 src/detector/train.py

echo "==> 3/3  adversarial test + adaptation loop"
python3 src/detector/mutate.py

echo
echo "Done. Start the demo with:  streamlit run app/main.py"
