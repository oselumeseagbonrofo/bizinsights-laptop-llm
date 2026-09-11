@echo off
title BizInsights LLM — Public Internet Tunnel
cd /d "%~dp0"

echo ======================================================================
echo   BizInsights AI — Public Internet Access (Cloudflare Tunnel)
echo ======================================================================
echo.

.\.venv\Scripts\python run_with_tunnel.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Tunnel exited with an error.
    pause
)
