@echo off
title BizInsights LLM — Mobile Assistant Server
cd /d "%~dp0"

echo ======================================================================
echo   BizInsights AI — Offline Customer Support Assistant
echo ======================================================================
:: Ensure port 8000 is clean before starting
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Freeing previous server process on port 8000 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    echo [INFO] Installing required dependencies...
    .\.venv\Scripts\pip install fastapi uvicorn httpx qrcode pillow
)

echo [INFO] Launching local network server...
.\.venv\Scripts\python server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server exited with an error.
    pause
)
