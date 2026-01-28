"""WhisperX-based audio transcription with speaker diarization.

Handles:
- Audio file transcription (any format ffmpeg supports)
- Word-level timestamps
- Speaker diarization (who said what)
- Korean language optimization
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.models.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)

# Module-level model cache
_whisper_model = None
_diarize_pipeline = None


def _get_whisper_model(model_name: str, device: str, compute_type: str):
    """Load or return cached WhisperX model."""
    global _whisper_model
    if _whisper_model is None:
        import whisperx
        logger.info(f"Loading WhisperX model: {model_name} (device={device}, compute={compute_type})")
        _whisper_model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language="ko",
        )
        logger.info("WhisperX model loaded.")
    return _whisper_model


def _get_diarize_pipeline(device: str, hf_token: str):
    """Load or return cached diarization pipeline."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        import whisperx
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set. Speaker diarization requires a HuggingFace token "
                "with access to pyannote/speaker-diarization-3.1. "
                "Set HF_TOKEN in .env to enable diarization."
            )
            return None
        logger.info("Loading diarization pipeline...")
        _diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        logger.info("Diarization pipeline loaded.")
    return _diarize_pipeline


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str = "ko",
    hf_token: str = "",
    enable_diarization: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Transcript:
    """Transcribe an audio file with optional speaker diarization.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, etc.)
        model_name: WhisperX model size
        device: "cpu" or "cuda"
        compute_type: "float32", "float16", or "int8"
        language: Language code (default "ko" for Korean)
        hf_token: HuggingFace token for diarization
        enable_diarization: Whether to run speaker diarization
        num_speakers: Exact number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)

    Returns:
        Transcript with speaker-labeled utterances
    """
    import whisperx

    audio_path = str(audio_path)
    logger.info(f"Starting transcription: {audio_path}")

    # Step 1: Load audio
    audio = whisperx.load_audio(audio_path)

    # Step 2: Transcribe
    model = _get_whisper_model(model_name, device, compute_type)
    result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4, language=language)
    detected_language = result.get("language", language)
    logger.info(f"Transcription complete. Language: {detected_language}, Segments: {len(result['segments'])}")

    # Step 3: Align word timestamps
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        logger.info("Word alignment complete.")
    except Exception as e:
        logger.warning(f"Word alignment failed (non-critical): {e}")

    # Step 4: Speaker diarization
    if enable_diarization:
        diarize_pipeline = _get_diarize_pipeline(device, hf_token)
        if diarize_pipeline is not None:
            try:
                diarize_kwargs = {}
                if num_speakers is not None:
                    diarize_kwargs["num_speakers"] = num_speakers
                if min_speakers is not None:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers

                diarize_segments = diarize_pipeline(audio_path, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("Speaker diarization complete.")
            except Exception as e:
                logger.warning(f"Diarization failed (non-critical): {e}")

    # Step 5: Build Transcript
    utterances = []
    for seg in result.get("segments", []):
        utterances.append(Utterance(
            speaker=seg.get("speaker", "UNKNOWN"),
            text=seg.get("text", "").strip(),
            start=seg.get("start"),
            end=seg.get("end"),
        ))

    # Merge consecutive utterances from same speaker
    merged = _merge_consecutive_speakers(utterances)

    transcript = Transcript(
        language=detected_language,
        utterances=merged,
    )

    logger.info(f"Transcript built: {len(merged)} utterances, {len(set(u.speaker for u in merged))} speakers")
    return transcript


def _merge_consecutive_speakers(utterances: list[Utterance]) -> list[Utterance]:
    """Merge consecutive utterances from the same speaker into one."""
    if not utterances:
        return []

    merged = [utterances[0].model_copy()]
    for u in utterances[1:]:
        if u.speaker == merged[-1].speaker:
            merged[-1].text += " " + u.text
            merged[-1].end = u.end
        else:
            merged.append(u.model_copy())
    return merged
