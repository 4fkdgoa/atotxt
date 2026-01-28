"""Audio-to-Text Transcription Service with Speaker Diarization.

Entry point for the FastAPI application.

Usage:
    python main.py
    # or
    uvicorn main:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /upload     - Upload audio file (async processing)
    GET  /task/{id}  - Check task status and results
    POST /transcribe - Upload and transcribe synchronously
    GET  /health     - Health check
    GET  /system     - System/GPU information
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="atotxt - Audio to Text",
    description="Voice transcription service with speaker diarization and AI summarization",
    version="1.0.0",
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    from app.services.gpu_detector import get_auto_config
    config = get_auto_config()
    logger.info(f"Device: {config.device}")
    logger.info(f"Whisper model: {config.whisper_model} ({config.compute_type})")
    logger.info(f"LLM model: {config.ollama_model}")
    logger.info(f"Server ready at http://{settings.host}:{settings.port}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )
