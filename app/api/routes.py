"""FastAPI routes for audio upload, transcription, and summarization."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import (
    MeetingType,
    SystemInfo,
    TaskStatus,
    TaskStatusResponse,
    TranscriptionResponse,
    UploadResponse,
)
from app.services.gpu_detector import detect_gpus, get_auto_config
from app.services.queue_manager import (
    create_task,
    get_task,
    process_audio_task,
)
from app.services.training_data import get_training_stats, export_for_finetuning

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".webm", ".mp4"}


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    enable_diarization: bool = Form(True, description="Enable speaker diarization"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers (optional)"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers (optional)"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers (optional)"),
    meeting_type: MeetingType = Form(MeetingType.GENERAL, description="Meeting type: 'general' or 'it_standup' for IT standup/sprint meetings"),
):
    """Upload an audio file for transcription and summarization.

    Returns a task ID immediately. Use GET /task/{task_id} to check progress.

    Meeting Types:
    - general: Standard meeting summarization (default)
    - it_standup: IT standup/sprint meeting with tech lead pattern, issue tracking, blockers, action items with assignees
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Validate file size
    if file.size and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {settings.max_upload_size_mb}MB",
        )

    # Save uploaded file
    task_id = create_task()
    upload_path = settings.upload_path / f"{task_id}{ext}"

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    logger.info(f"[{task_id}] Uploaded: {file.filename} ({upload_path})")

    # Start async processing (non-blocking)
    asyncio.create_task(
        process_audio_task(
            task_id=task_id,
            audio_path=upload_path,
            enable_diarization=enable_diarization,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            meeting_type=meeting_type,
        )
    )

    return UploadResponse(task_id=task_id)


@router.get("/task/{task_id}", response_model=TranscriptionResponse)
async def get_task_status(task_id: str):
    """Check the status and results of a transcription task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/system", response_model=SystemInfo)
async def system_info():
    """Return system hardware and model configuration."""
    config = get_auto_config()
    gpus = detect_gpus()
    return SystemInfo(
        device=config.device,
        whisper_model=config.whisper_model,
        compute_type=config.compute_type,
        ollama_model=config.ollama_model,
        gpus=gpus,
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_sync(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    enable_diarization: bool = Form(True, description="Enable speaker diarization"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    meeting_type: MeetingType = Form(MeetingType.GENERAL, description="Meeting type: 'general' or 'it_standup'"),
):
    """Upload and transcribe synchronously. Blocks until complete.

    Use this for small files or when you need immediate results.
    For large files, use POST /upload instead.

    Meeting Types:
    - general: Standard meeting summarization (default)
    - it_standup: IT standup/sprint meeting with specialized analysis
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    task_id = create_task()
    upload_path = settings.upload_path / f"{task_id}{ext}"

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Process synchronously (blocking)
    await process_audio_task(
        task_id=task_id,
        audio_path=upload_path,
        enable_diarization=enable_diarization,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        meeting_type=meeting_type,
    )

    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=500, detail="Task processing failed unexpectedly")

    if task.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=task.error or "Processing failed")

    return task


@router.get("/training/stats")
async def training_stats():
    """Get statistics about collected training data for fine-tuning."""
    stats = get_training_stats()
    return {
        **stats,
        "message": "파인튜닝 준비 완료!" if stats["ready_for_finetuning"] else f"데이터 수집 중... (최소 50개 필요, 현재 {stats['total']}개)",
    }


@router.post("/training/export")
async def export_training_data(
    meeting_type: Optional[MeetingType] = Form(None, description="Filter by meeting type (None for all)"),
):
    """Export collected training data as JSONL for fine-tuning."""
    from datetime import datetime

    filename = f"training_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_path = settings.training_data_path / filename

    export_for_finetuning(
        output_path=str(output_path),
        meeting_type=meeting_type,
    )

    return {
        "file": str(output_path),
        "message": "내보내기 완료! 이 파일로 파인튜닝하세요.",
    }
