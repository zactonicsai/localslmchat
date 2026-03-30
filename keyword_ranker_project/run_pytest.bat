@echo off
setlocal
cd /d %~dp0
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -v
endlocal
