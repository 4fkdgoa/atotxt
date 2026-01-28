#!/bin/bash
# Linux Installation Script for atotxt
set -e

echo "=== atotxt - Audio to Text Service Installer ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PY_VERSION=$(python3 --version)
echo "[OK] $PY_VERSION"

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARN] ffmpeg not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    else
        echo "[ERROR] Cannot auto-install ffmpeg. Please install manually."
        exit 1
    fi
else
    echo "[OK] ffmpeg found"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "[INFO] venv already exists, skipping"
else
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check GPU
echo ""
echo "Checking GPU availability..."
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem // (1024**2)
        print(f'[OK] GPU: {name} ({vram}MB)')
    else:
        print('[INFO] No CUDA GPU detected. Running in CPU mode.')
except ImportError:
    print('[INFO] PyTorch not installed with CUDA. CPU mode.')
"

# Check Ollama
echo ""
if ! command -v ollama &> /dev/null; then
    echo "[WARN] Ollama not found."
    echo "       Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "       Then: ollama pull qwen2.5:14b"
else
    echo "[OK] Ollama found"
fi

# Create .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "[INFO] Created .env from .env.example"
    echo "       Edit .env to set HF_TOKEN for speaker diarization."
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and set HF_TOKEN (for speaker diarization)"
echo "  2. Start Ollama:  ollama serve"
echo "  3. Pull LLM:      ollama pull qwen2.5:14b"
echo "  4. Start server:   python main.py"
echo "  5. Test:           curl -F 'file=@sample.wav' http://localhost:8000/upload"
echo ""
