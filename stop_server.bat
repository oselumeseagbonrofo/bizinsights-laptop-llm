@echo off
title Stop BizInsights LLM Server
cd /d "%~dp0"

echo Stopping BizInsights model server, python web server, and llama engine...

:: Kill llama-server engine
taskkill /F /IM llama-server.exe >nul 2>&1

:: Free port 8000 if occupied by any lingering process
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo Server stopped and port 8000 freed cleanly.
timeout /t 2 >nul
