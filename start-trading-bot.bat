@echo off
REM Launch the paper-trading bot (canonical launcher: start_paper_trading.py).
REM cd to the repo dir so relative paths (config, logs, learned_params.json) resolve.
cd /d "%~dp0"
python start_paper_trading.py
echo.
echo === Paper trading process exited ===
pause
