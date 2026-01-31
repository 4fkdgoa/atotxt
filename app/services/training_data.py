"""Training data collection for fine-tuning.

Saves transcripts and summaries in a format ready for fine-tuning.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Union

from app.core.config import settings
from app.models.schemas import (
    MeetingType,
    Transcript,
    SummaryResult,
    ITSummaryResult,
)

logger = logging.getLogger(__name__)


def save_training_data(
    transcript: Transcript,
    summary: Union[SummaryResult, ITSummaryResult],
    meeting_type: MeetingType,
    task_id: str,
) -> Path | None:
    """Save transcript and summary as training data.

    Args:
        transcript: The transcribed audio
        summary: The generated summary
        meeting_type: Type of meeting (general or it_standup)
        task_id: Unique task identifier

    Returns:
        Path to saved file, or None if saving is disabled
    """
    if not settings.save_training_data:
        return None

    try:
        # Create directory for meeting type
        type_dir = settings.training_data_path / meeting_type.value
        type_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{task_id}.json"
        filepath = type_dir / filename

        # Prepare training data format
        training_record = {
            "id": task_id,
            "timestamp": datetime.now().isoformat(),
            "meeting_type": meeting_type.value,
            "input": {
                "transcript": transcript.to_dialogue_text(),
                "language": transcript.language,
                "utterance_count": len(transcript.utterances),
                "speakers": list(set(u.speaker for u in transcript.utterances)),
            },
            "output": summary.model_dump(),
            # For fine-tuning format (input -> output)
            "training_format": {
                "instruction": _get_instruction(meeting_type),
                "input": transcript.to_dialogue_text(),
                "output": json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            },
        }

        # Save
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(training_record, f, ensure_ascii=False, indent=2)

        logger.info(f"Training data saved: {filepath}")

        # Log collection progress
        _log_collection_progress(meeting_type)

        return filepath

    except Exception as e:
        logger.error(f"Failed to save training data: {e}")
        return None


def _get_instruction(meeting_type: MeetingType) -> str:
    """Get instruction text for the training format."""
    if meeting_type == MeetingType.IT_STANDUP:
        return (
            "다음 IT 개발팀 스탠드업/스프린트 회의 녹취록을 분석하여 "
            "기술 개요, 주제별 진행 상황, 블로커, 액션 아이템을 JSON 형식으로 정리하세요. "
            "기술 용어는 영어 원문을 유지하세요."
        )
    else:
        return (
            "다음 회의 녹취록을 분석하여 주제별 요약과 액션 아이템을 JSON 형식으로 정리하세요."
        )


def _log_collection_progress(meeting_type: MeetingType):
    """Log how many training samples have been collected."""
    try:
        type_dir = settings.training_data_path / meeting_type.value
        if type_dir.exists():
            count = len(list(type_dir.glob("*.json")))

            # Milestones
            milestones = [50, 100, 250, 500, 1000]
            for milestone in milestones:
                if count == milestone:
                    logger.info(f"🎉 [{meeting_type.value}] {milestone}개 달성! 파인튜닝 준비 완료.")
                    break
            else:
                if count % 10 == 0:  # Log every 10
                    logger.info(f"[{meeting_type.value}] 학습 데이터: {count}개 수집됨")
    except Exception:
        pass


def get_training_stats() -> dict:
    """Get statistics about collected training data."""
    stats = {
        "total": 0,
        "by_type": {},
        "ready_for_finetuning": False,
    }

    if not settings.training_data_path.exists():
        return stats

    for meeting_type in MeetingType:
        type_dir = settings.training_data_path / meeting_type.value
        if type_dir.exists():
            count = len(list(type_dir.glob("*.json")))
            stats["by_type"][meeting_type.value] = count
            stats["total"] += count

    stats["ready_for_finetuning"] = stats["total"] >= 50
    return stats


def export_for_finetuning(
    output_path: str = "training_data_export.jsonl",
    meeting_type: MeetingType | None = None,
) -> Path:
    """Export training data in JSONL format for fine-tuning.

    Args:
        output_path: Output file path
        meeting_type: Filter by meeting type (None for all)

    Returns:
        Path to exported file
    """
    output_file = Path(output_path)
    records = []

    types_to_export = [meeting_type] if meeting_type else list(MeetingType)

    for mtype in types_to_export:
        type_dir = settings.training_data_path / mtype.value
        if not type_dir.exists():
            continue

        for json_file in type_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract training format
                    if "training_format" in data:
                        records.append(data["training_format"])
            except Exception as e:
                logger.warning(f"Failed to read {json_file}: {e}")

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(records)} training records to {output_file}")
    return output_file
