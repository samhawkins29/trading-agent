@echo off
REM Pre-flight check: print system status (market hours, Alpaca connection,
REM risk state, strategy weights) without starting the trader.
cd /d "%~dp0"
python start_paper_trading.py --status
echo.
echo === Status check complete ===
pause
