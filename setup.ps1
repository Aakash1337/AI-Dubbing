# AI Dubbing — one-time environment setup (Windows / PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cpu     # CPU-only torch
#
# Creates a project-local .venv, installs the CUDA (or CPU) PyTorch build, then
# the rest of requirements (including chatterbox-tts). Re-runnable.

param(
    [switch]$Cpu,
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu124"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv) ..." -ForegroundColor Cyan
    python -m venv .venv
}

$py = Join-Path $here ".venv\Scripts\python.exe"

Write-Host "Upgrading pip ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip

if ($Cpu) {
    Write-Host "Installing CPU PyTorch ..." -ForegroundColor Cyan
    & $py -m pip install torch torchaudio
} else {
    Write-Host "Installing CUDA PyTorch from $CudaIndex ..." -ForegroundColor Cyan
    & $py -m pip install torch torchaudio --index-url $CudaIndex
}

Write-Host "Installing remaining requirements ..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Verify GPU + model wiring with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe dub.py --dry-run" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe dub.py --only 01 --max-cues 15   # fast preview" -ForegroundColor Green
