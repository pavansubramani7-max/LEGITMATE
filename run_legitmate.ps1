$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment is missing. From this folder run: py -m venv .venv; then .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Write-Host 'Port 8000 already has a listener. Open http://127.0.0.1:8000 or stop that server before starting another.' -ForegroundColor Yellow
    exit 1
}

Set-Location -LiteralPath $projectRoot
$env:LEGITMATE_BIND_HOST = '127.0.0.1'
Write-Host 'Starting LEGITMATE at http://127.0.0.1:8000 — press Ctrl+C to stop.' -ForegroundColor Cyan
& $python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
