@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Tao moi truong Python...
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Da cai dat xong. De kiem tra bang du lieu minh hoa, chay:
echo python train.py --data data/sample_demo.csv
echo Sau do chay:
echo streamlit run app.py
pause
