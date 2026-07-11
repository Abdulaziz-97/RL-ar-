@echo off
echo ============================================
echo Arabic Reasoning RLVR - Install
echo ============================================
echo.
echo Step 1: Create virtual environment
python -m venv .venv
call .venv\Scripts\activate.bat

echo Step 2: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo.
echo Done! Run 'run_sft.bat' or 'run_grpo.bat' to start training.
echo.
pause
