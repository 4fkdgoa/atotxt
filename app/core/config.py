"""Application configuration with environment variable support."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # WhisperX
    whisper_model: str = "auto"  # auto, tiny, base, small, medium, large-v2, large-v3
    whisper_language: str = "ko"
    whisper_device: str = "auto"  # auto, cpu, cuda
    whisper_compute_type: str = "auto"  # auto, float16, int8, float32

    # Upload
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 500

    # Worker
    worker_concurrency: int = 2

    # HuggingFace token (required for speaker diarization)
    hf_token: str = ""

    # Training data collection (for fine-tuning)
    save_training_data: bool = False
    training_data_dir: str = "./training_data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def training_data_path(self) -> Path:
        path = Path(self.training_data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
