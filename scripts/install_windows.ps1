# Windows Server Installation Script for atotxt
# Run as Administrator in PowerShell

Write-Host "=== atotxt - Audio to Text Service Installer ===" -ForegroundColor Cyan
Write-Host ""

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

$pyVersion = python --version 2>&1
Write-Host "[OK] $pyVersion" -ForegroundColor Green

# Check ffmpeg
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "[WARN] ffmpeg not found. Installing via winget..." -ForegroundColor Yellow
    winget install Gyan.FFmpeg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] ffmpeg install failed. Download from https://ffmpeg.org/download.html" -ForegroundColor Red
        Write-Host "        Add ffmpeg to your PATH after installation." -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] ffmpeg found" -ForegroundColor Green
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Cyan

if (Test-Path "venv") {
    Write-Host "[INFO] venv already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv venv
}

# Activate venv
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

# Check GPU
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Cyan
$gpuCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>&1

if ($gpuCheck -match "CUDA: True") {
    Write-Host "[OK] GPU detected: $gpuCheck" -ForegroundColor Green
    Write-Host "[INFO] For GPU acceleration, install PyTorch with CUDA:" -ForegroundColor Yellow
    Write-Host "       pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] No GPU detected. Running in CPU mode." -ForegroundColor Yellow
    Write-Host "[INFO] CPU mode works fine for batch processing." -ForegroundColor Yellow
}

# Check Ollama
Write-Host ""
Write-Host "Checking Ollama..." -ForegroundColor Cyan
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "[WARN] Ollama not found." -ForegroundColor Yellow
    Write-Host "       Download from: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "       After install: ollama pull qwen2.5:14b" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Ollama found" -ForegroundColor Green
    Write-Host "[INFO] Pull a model: ollama pull qwen2.5:14b" -ForegroundColor Yellow
}

# Create .env if not exists
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "[INFO] Created .env from .env.example" -ForegroundColor Yellow
    Write-Host "       Edit .env to set HF_TOKEN for speaker diarization." -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env and set HF_TOKEN (for speaker diarization)"
Write-Host "  2. Start Ollama:  ollama serve"
Write-Host "  3. Pull LLM:      ollama pull qwen2.5:14b"
Write-Host "  4. Start server:   python main.py"
Write-Host "  5. Test:           curl -F 'file=@sample.wav' http://localhost:8000/upload"
Write-Host ""
