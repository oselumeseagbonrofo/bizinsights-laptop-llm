@echo off
title BizInsights LLM Server (Custom Domain Active)
cd /d "%~dp0"

echo ======================================================================
echo   BizInsights AI -- Local Model Server
echo   Cloudflare Named Tunnel is running as a Windows Background Service!
echo ======================================================================
echo.

:: Ensure port 8000 is clean before starting
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Freeing previous server process on port 8000 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: Ensure virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Setting up virtual environment...
    python -m venv .venv
    .\.venv\Scripts\pip install fastapi uvicorn httpx qrcode pillow
)

:: Check if Cloudflared Windows Service is running
sc query Cloudflared | findstr /I "RUNNING" >nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Cloudflare Tunnel Service is RUNNING and connected to your domain.
) else (
    echo [WARNING] Cloudflare Tunnel service is not running. Attempting to start it...
    net start Cloudflared
)

echo.
echo [INFO] Starting LLM inference server on port 8000...
echo All traffic to your custom domain is automatically routed to this server!
echo.
.\.venv\Scripts\python server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server encountered an error.
    pause
)
