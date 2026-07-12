#!/usr/bin/env bash
# Reproduziert die Ergebnisse auf Linux: venv, Installation, Training, Evaluation.
set -e
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python train.py --headless --curriculum --auto-aim-red --seed 42 --max-games 1500
python eval.py --episodes 100 --headless --seed 123 --auto-aim-red --blue-error 0.0 --blue-mode mixed
