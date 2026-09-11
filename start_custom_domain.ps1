# BizInsights LLM — Local Server with Custom Domain Tunnel
$Host.UI.RawUI.WindowTitle = "BizInsights LLM Server (Custom Domain Active)"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  BizInsights AI — Local Model Server" -ForegroundColor Cyan
Write-Host "  Cloudflare Named Tunnel is running as a Windows Service" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$svc = Get-Service -Name "Cloudflared" -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -eq "Running") {
    Write-Host "[OK] Cloudflare Tunnel Service is RUNNING and connected." -ForegroundColor Green
} else {
    Write-Host "[INFO] Starting Cloudflare Tunnel Service..." -ForegroundColor Yellow
    Start-Service -Name "Cloudflared" -ErrorAction SilentlyContinue
}

Write-Host "[INFO] Launching local model server on port 8000..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" server.py
