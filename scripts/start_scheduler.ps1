Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$scheduler = Join-Path $projectRoot "run_scheduler.py"
$logDir = Join-Path $projectRoot "logs"
$logFile = Join-Path $logDir "scheduler.log"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Set-Location $projectRoot

"[$(Get-Date -Format s)] Starting PredictionMLBot scheduler" | Out-File -FilePath $logFile -Append -Encoding utf8
& $pythonExe -u $scheduler *>> $logFile
