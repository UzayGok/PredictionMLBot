@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "PYTHONW_EXE=%PROJECT_ROOT%\.venv\Scripts\pythonw.exe"
set "SCHEDULER=%PROJECT_ROOT%\run_scheduler.py"
cd /d "%PROJECT_ROOT%"
start "PredictionMLBotScheduler" /min "%PYTHONW_EXE%" "%SCHEDULER%"
exit /b 0
