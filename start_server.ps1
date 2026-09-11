# BizInsights LLM — Local Network Mobile Assistant Server
$Host.UI.RawUI.WindowTitle = "BizInsights LLM — Mobile Assistant Server"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  BizInsights AI — Offline Customer Support Assistant" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[INFO] Creating Python virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[INFO] Installing required dependencies..." -ForegroundColor Yellow
    .\.venv\Scripts\pip install fastapi uvicorn httpx qrcode pillow
}

Write-Host "[INFO] Launching local network server..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" server.py
