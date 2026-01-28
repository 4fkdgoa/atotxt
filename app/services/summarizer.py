"""Ollama-based transcript summarization and topic classification.

Takes a speaker-diarized transcript and produces:
- Topic-based grouping
- Per-topic summaries
- Action items
- Overall summary
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from app.models.schemas import SummaryResult, TopicSummary, Transcript

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """당신은 회의록 분석 전문가입니다.
주어진 대화록을 분석하여 반드시 아래 JSON 형식으로만 응답하세요.
다른 텍스트 없이 JSON만 출력하세요."""

SUMMARY_USER_PROMPT_TEMPLATE = """다음은 회의/통화 대화록입니다. 화자별로 구분되어 있습니다.

대화록:
{transcript}

위 대화록을 분석하여 다음을 수행하세요:

1. 전체 대화를 주제별로 분류하세요
2. 각 주제별 핵심 내용을 요약하세요
3. 각 주제에 참여한 화자를 표시하세요
4. 전체 대화의 종합 요약을 작성하세요
5. 후속 조치 사항(Action Items)이 있다면 추출하세요

반드시 아래 JSON 형식으로만 응답하세요:

{{
  "overall_summary": "전체 대화 종합 요약 (2~3문장)",
  "topics": [
    {{
      "topic": "주제명",
      "summary": "주제 요약",
      "speakers": ["SPEAKER_00", "SPEAKER_01"]
    }}
  ],
  "action_items": ["후속 조치 사항 1", "후속 조치 사항 2"]
}}"""


async def summarize_transcript(
    transcript: Transcript,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:14b",
    timeout: float = 300.0,
) -> SummaryResult:
    """Summarize a transcript using Ollama LLM.

    Args:
        transcript: Speaker-diarized transcript
        ollama_base_url: Ollama API base URL
        model: Ollama model name
        timeout: Request timeout in seconds

    Returns:
        SummaryResult with topics, summaries, and action items
    """
    dialogue_text = transcript.to_dialogue_text()

    if not dialogue_text.strip():
        logger.warning("Empty transcript, returning empty summary.")
        return SummaryResult(overall_summary="대화 내용이 없습니다.")

    prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(transcript=dialogue_text)

    logger.info(f"Requesting summary from Ollama ({model})...")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": SUMMARY_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            raw_text = result.get("response", "")

        logger.info(f"Ollama response received ({len(raw_text)} chars)")
        return _parse_summary(raw_text)

    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {ollama_base_url}. Is Ollama running?")
        return SummaryResult(
            overall_summary="요약 실패: Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요."
        )
    except httpx.TimeoutException:
        logger.error("Ollama request timed out.")
        return SummaryResult(overall_summary="요약 실패: 요청 시간 초과")
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return SummaryResult(overall_summary=f"요약 실패: {str(e)}")


def _parse_summary(raw_text: str) -> SummaryResult:
    """Parse LLM output into structured SummaryResult.

    Handles cases where the LLM wraps JSON in markdown code blocks
    or includes extra text around the JSON.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object in the text
    json_start = text.find("{")
    json_end = text.rfind("}") + 1

    if json_start == -1 or json_end <= json_start:
        logger.warning("No JSON found in LLM response, using raw text as summary.")
        return SummaryResult(overall_summary=text[:500])

    json_str = text[json_start:json_end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from LLM response: {e}")
        return SummaryResult(overall_summary=text[:500])

    topics = []
    for t in data.get("topics", []):
        topics.append(TopicSummary(
            topic=t.get("topic", ""),
            summary=t.get("summary", ""),
            speakers=t.get("speakers", []),
        ))

    return SummaryResult(
        overall_summary=data.get("overall_summary", ""),
        topics=topics,
        action_items=data.get("action_items", []),
    )
