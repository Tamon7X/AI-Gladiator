@echo off
REM Reproduziert die Ergebnisse auf Windows: venv, Installation, Training, Evaluation.
cd /d "%~dp0\.."

python -m venv .venv
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

python train.py --headless --curriculum --auto-aim-red --seed 42 --max-games 1500
python eval.py --episodes 100 --headless --seed 123 --auto-aim-red --blue-error 0.0 --blue-mode mixed
