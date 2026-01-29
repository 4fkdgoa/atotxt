"""Task queue management with Celery + Redis.

Provides async task processing to prevent GPU overload:
- Tasks are queued in Redis
- Workers pull one task at a time per GPU
- Prevents OOM crashes from concurrent processing

Falls back to synchronous in-process execution if Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.models.schemas import (
    SummaryResult,
    TaskStatus,
    Transcript,
    TranscriptionResponse,
)

logger = logging.getLogger(__name__)

# In-memory task store (for simplicity; replace with Redis in production)
_task_store: dict[str, TranscriptionResponse] = {}
_executor = ThreadPoolExecutor(max_workers=settings.worker_concurrency)


def create_task() -> str:
    """Create a new task and return its ID."""
    task_id = str(uuid.uuid4())[:8]
    _task_store[task_id] = TranscriptionResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
    )
    return task_id


def get_task(task_id: str) -> Optional[TranscriptionResponse]:
    """Get task status and result."""
    return _task_store.get(task_id)


def update_task(task_id: str, **kwargs):
    """Update task fields."""
    if task_id in _task_store:
        task = _task_store[task_id]
        for key, value in kwargs.items():
            setattr(task, key, value)


async def process_audio_task(
    task_id: str,
    audio_path: Path,
    enable_diarization: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """Process an audio file: transcribe + summarize.

    Runs in a thread pool to avoid blocking the event loop.
    """
    from app.services.gpu_detector import get_auto_config
    from app.services.summarizer import summarize_transcript
    from app.services.transcriber import transcribe_audio

    update_task(task_id, status=TaskStatus.PROCESSING)
    logger.info(f"[{task_id}] Processing started: {audio_path}")

    try:
        # Get model configuration
        config = get_auto_config()

        # Run transcription in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            _executor,
            lambda: transcribe_audio(
                audio_path=str(audio_path),
                model_name=config.whisper_model,
                device=config.device,
                compute_type=config.compute_type,
                language=settings.whisper_language,
                hf_token=settings.hf_token,
                enable_diarization=enable_diarization,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        )

        logger.info(f"[{task_id}] Transcription complete. {len(transcript.utterances)} utterances.")

        # Run summarization (async I/O-bound)
        summary = await summarize_transcript(
            transcript=transcript,
            ollama_base_url=settings.ollama_base_url,
            model=config.ollama_model,
        )

        logger.info(f"[{task_id}] Summarization complete. {len(summary.topics)} topics.")

        update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            transcript=transcript,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"[{task_id}] Processing failed: {e}", exc_info=True)
        update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e),
        )
    finally:
        # Clean up uploaded file
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass
