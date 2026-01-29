"""Tests for GPU detection and model selection."""

from app.models.schemas import GPUInfo
from app.services.gpu_detector import select_models


def test_cpu_only():
    gpus = [GPUInfo(name="CPU", vram_total_mb=0, vram_free_mb=0, cuda_available=False)]
    config = select_models(gpus)
    assert config.device == "cpu"
    assert config.whisper_model == "base"
    assert config.compute_type == "float32"


def test_low_vram_gpu():
    """GTX 1660 class (6GB)."""
    gpus = [GPUInfo(name="GTX 1660", vram_total_mb=6000, vram_free_mb=5500, cuda_available=True)]
    config = select_models(gpus)
    assert config.device == "cuda"
    assert config.whisper_model == "small"
    assert config.compute_type == "int8"


def test_mid_vram_gpu():
    """RTX 3060 class (8-12GB)."""
    gpus = [GPUInfo(name="RTX 3060", vram_total_mb=12000, vram_free_mb=11000, cuda_available=True)]
    config = select_models(gpus)
    assert config.device == "cuda"
    assert config.whisper_model == "medium"
    assert config.compute_type == "int8"


def test_high_vram_gpu():
    """RTX 5070 Ti class (16GB+)."""
    gpus = [GPUInfo(name="RTX 5070 Ti", vram_total_mb=16384, vram_free_mb=15000, cuda_available=True)]
    config = select_models(gpus)
    assert config.device == "cuda"
    assert config.whisper_model == "large-v3"
    assert config.compute_type == "float16"
    assert config.ollama_model == "qwen2.5:14b"


def test_user_override():
    gpus = [GPUInfo(name="CPU", vram_total_mb=0, vram_free_mb=0, cuda_available=False)]
    config = select_models(gpus, preferred_whisper="medium", preferred_compute="int8")
    assert config.whisper_model == "medium"
    assert config.compute_type == "int8"


def test_multiple_gpus_picks_best():
    gpus = [
        GPUInfo(name="GTX 1660", vram_total_mb=6000, vram_free_mb=5500, cuda_available=True),
        GPUInfo(name="RTX 5070 Ti", vram_total_mb=16384, vram_free_mb=15000, cuda_available=True),
    ]
    config = select_models(gpus)
    assert config.whisper_model == "large-v3"
