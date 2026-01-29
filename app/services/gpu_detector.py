"""GPU auto-detection and model selection.

Detects available GPUs and their VRAM to automatically choose
the optimal Whisper model size and compute type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.models.schemas import GPUInfo

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Selected model configuration based on hardware."""
    device: str               # "cuda" or "cpu"
    whisper_model: str        # e.g. "large-v3", "small", "base"
    compute_type: str         # e.g. "float16", "int8", "float32"
    ollama_model: str         # e.g. "qwen2.5:14b", "qwen2.5:7b"
    whisper_vram_mb: int      # Estimated VRAM usage for Whisper
    llm_vram_mb: int          # Estimated VRAM usage for LLM


# VRAM requirements (approximate, Q4 quantization for LLMs)
WHISPER_VRAM = {
    "tiny": 400,
    "base": 500,
    "small": 1000,
    "medium": 2500,
    "large-v2": 4000,
    "large-v3": 4000,
}

LLM_VRAM = {
    "qwen2.5:7b": 5000,
    "llama3:8b": 5500,
    "qwen2.5:14b": 9500,
    "mistral-small:22b": 13000,
    "qwen2.5:32b": 19000,
}


def detect_gpus() -> list[GPUInfo]:
    """Detect available NVIDIA GPUs and their VRAM."""
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_mem // (1024 * 1024)
                free_mb = total_mb  # Approximate; actual free depends on usage
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(i)
                    free_mb = free_bytes // (1024 * 1024)
                except Exception:
                    pass
                gpus.append(GPUInfo(
                    name=props.name,
                    vram_total_mb=total_mb,
                    vram_free_mb=free_mb,
                    cuda_available=True,
                ))
                logger.info(f"GPU {i}: {props.name} ({total_mb}MB total, {free_mb}MB free)")
    except ImportError:
        logger.info("PyTorch not available or no CUDA support. Using CPU mode.")
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}. Falling back to CPU.")

    if not gpus:
        gpus.append(GPUInfo(
            name="CPU",
            vram_total_mb=0,
            vram_free_mb=0,
            cuda_available=False,
        ))
    return gpus


def select_models(
    gpus: list[GPUInfo],
    preferred_whisper: str = "auto",
    preferred_compute: str = "auto",
    preferred_ollama: str = "",
) -> ModelConfig:
    """Select optimal models based on available hardware.

    Strategy:
    - GPU with >=16GB: Whisper large-v3 + qwen2.5:14b (both fit)
    - GPU with >=8GB:  Whisper medium + qwen2.5:7b
    - GPU with >=4GB:  Whisper small + LLM on CPU via Ollama
    - CPU only:        Whisper base + qwen2.5:7b (Ollama CPU)
    """
    best_gpu = max(gpus, key=lambda g: g.vram_total_mb)
    vram = best_gpu.vram_total_mb
    has_cuda = best_gpu.cuda_available

    if has_cuda and vram >= 16000:
        # High-end: RTX 5070 Ti / RTX 4090 / A100 class
        whisper = "large-v3"
        compute = "float16"
        ollama = "qwen2.5:14b"
        device = "cuda"
    elif has_cuda and vram >= 8000:
        # Mid-range: RTX 3060 / RTX 4060 class
        whisper = "medium"
        compute = "int8"
        ollama = "qwen2.5:7b"
        device = "cuda"
    elif has_cuda and vram >= 4000:
        # Low-end: GTX 1660 class
        whisper = "small"
        compute = "int8"
        ollama = "llama3:8b"
        device = "cuda"
    else:
        # CPU only
        whisper = "base"
        compute = "float32"
        ollama = "qwen2.5:7b"
        device = "cpu"

    # Apply user overrides
    if preferred_whisper != "auto":
        whisper = preferred_whisper
    if preferred_compute != "auto":
        compute = preferred_compute
    if preferred_ollama:
        ollama = preferred_ollama

    whisper_vram = WHISPER_VRAM.get(whisper, 2000)
    llm_vram = LLM_VRAM.get(ollama, 5000)

    config = ModelConfig(
        device=device,
        whisper_model=whisper,
        compute_type=compute,
        ollama_model=ollama,
        whisper_vram_mb=whisper_vram,
        llm_vram_mb=llm_vram,
    )

    logger.info(
        f"Model selection: device={device}, whisper={whisper} ({compute}), "
        f"llm={ollama}, VRAM needed={whisper_vram + llm_vram}MB"
    )

    if has_cuda and (whisper_vram + llm_vram) > vram:
        logger.warning(
            f"Combined VRAM ({whisper_vram + llm_vram}MB) exceeds available ({vram}MB). "
            f"Whisper and LLM will NOT run simultaneously. "
            f"Consider using a smaller model or sequential processing."
        )

    return config


def get_auto_config() -> ModelConfig:
    """Detect hardware and return optimal model configuration."""
    from app.core.config import settings

    gpus = detect_gpus()
    return select_models(
        gpus,
        preferred_whisper=settings.whisper_model,
        preferred_compute=settings.whisper_compute_type,
        preferred_ollama=settings.ollama_model if settings.ollama_model != "qwen2.5:14b" else "",
    )
