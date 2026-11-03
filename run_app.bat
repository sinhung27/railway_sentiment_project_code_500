@echo off
call .venv\Scripts\activate.bat
python train.py
if errorlevel 1 pause & exit /b 1
streamlit run app.py
