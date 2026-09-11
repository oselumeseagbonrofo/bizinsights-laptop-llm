# BizInsights LLM — Public Internet Tunnel
$Host.UI.RawUI.WindowTitle = "BizInsights LLM — Public Internet Tunnel"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  BizInsights AI — Public Internet Access (Cloudflare Tunnel)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

& ".\.venv\Scripts\python.exe" run_with_tunnel.py
