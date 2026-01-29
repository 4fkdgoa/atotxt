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

from app.models.schemas import (
    SummaryResult, TopicSummary, Transcript,
    ITSummaryResult, ITTopicSummary, TechOverview, Blocker, ActionItem
)

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


# ============================================================
# IT 스탠드업/스프린트 회의 특화 프롬프트
# ============================================================

IT_STANDUP_SYSTEM_PROMPT = """당신은 IT 개발팀 회의록 분석 전문가입니다.
스탠드업, 스프린트 리뷰, 기술 회의를 분석합니다.

핵심 규칙:
- 기술 용어는 영어 원문 유지 (API, PR, CI/CD, deploy, merge, hotfix 등)
- SPEAKER_00은 대체로 회의 진행자/기술 리드입니다
- 이슈 번호, PR 번호가 언급되면 반드시 포함하세요

반드시 JSON 형식으로만 응답하세요."""

IT_STANDUP_USER_PROMPT_TEMPLATE = """다음은 IT 개발팀 스탠드업/스프린트 회의 녹취록입니다.

녹취록:
{transcript}

---

위 회의를 분석하여 다음 JSON 형식으로 정리하세요:

{{
  "overall_summary": "회의 전체 요약 (무엇을 논의했고, 핵심 결정사항은 무엇인지)",

  "tech_overview": {{
    "description": "SPEAKER_00(기술 리드)가 언급한 기술적 컨텍스트/아키텍처 관련 내용",
    "technologies": ["언급된 기술 스택 목록"]
  }},

  "topics": [
    {{
      "topic": "주제명 (예: 로그인 API 버그 수정)",
      "summary": "진행 상황 및 내용 요약",
      "speakers": ["참여 화자"],
      "status": "done | in_progress | blocked | planned",
      "issue_refs": ["이슈/PR 번호가 있으면 기재"]
    }}
  ],

  "blockers": [
    {{
      "issue": "블로커 설명",
      "owner": "담당 화자",
      "needs": "필요한 지원/리소스"
    }}
  ],

  "action_items": [
    {{
      "task": "할 일 내용",
      "assignee": "담당자 (SPEAKER_XX 또는 언급된 이름)",
      "deadline": "언급된 기한 (없으면 null)",
      "priority": "high | medium | low"
    }}
  ],

  "decisions": ["회의에서 결정된 사항들"],

  "next_steps": ["다음 단계/다음 회의까지 해야 할 것들"]
}}

참고:
- status 판단 기준: "완료/머지/배포됨" → done, "하는 중/진행 중" → in_progress, "막혀있음/대기 중" → blocked, "예정/할 예정" → planned
- 블로커가 없으면 빈 배열
- 기술 용어는 번역하지 말고 원문 유지"""


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


# ============================================================
# IT Standup/Sprint Meeting Summarization
# ============================================================

async def summarize_it_standup(
    transcript: Transcript,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:14b",
    timeout: float = 300.0,
) -> ITSummaryResult:
    """Summarize an IT standup/sprint meeting transcript.

    Optimized for:
    - Status updates (what was done, what's blocked)
    - Technical discussions (SPEAKER_00 as tech lead)
    - Mixed Korean/English with tech terms
    - Issue/PR tracking

    Args:
        transcript: Speaker-diarized transcript
        ollama_base_url: Ollama API base URL
        model: Ollama model name
        timeout: Request timeout in seconds

    Returns:
        ITSummaryResult with structured meeting analysis
    """
    dialogue_text = transcript.to_dialogue_text()

    if not dialogue_text.strip():
        logger.warning("Empty transcript, returning empty summary.")
        return ITSummaryResult(overall_summary="회의 내용이 없습니다.")

    prompt = IT_STANDUP_USER_PROMPT_TEMPLATE.format(transcript=dialogue_text)

    logger.info(f"Requesting IT standup summary from Ollama ({model})...")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": IT_STANDUP_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more consistent structure
                        "num_predict": 4096,  # More tokens for detailed output
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            raw_text = result.get("response", "")

        logger.info(f"Ollama IT summary response received ({len(raw_text)} chars)")
        return _parse_it_summary(raw_text)

    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {ollama_base_url}.")
        return ITSummaryResult(
            overall_summary="요약 실패: Ollama 서버에 연결할 수 없습니다."
        )
    except httpx.TimeoutException:
        logger.error("Ollama request timed out.")
        return ITSummaryResult(overall_summary="요약 실패: 요청 시간 초과")
    except Exception as e:
        logger.error(f"IT summarization failed: {e}")
        return ITSummaryResult(overall_summary=f"요약 실패: {str(e)}")


def _parse_it_summary(raw_text: str) -> ITSummaryResult:
    """Parse LLM output into structured ITSummaryResult."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Find JSON
    json_start = text.find("{")
    json_end = text.rfind("}") + 1

    if json_start == -1 or json_end <= json_start:
        logger.warning("No JSON found in IT summary response.")
        return ITSummaryResult(overall_summary=text[:500])

    json_str = text[json_start:json_end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse IT summary JSON: {e}")
        return ITSummaryResult(overall_summary=text[:500])

    # Parse tech_overview
    tech_overview = None
    if "tech_overview" in data and data["tech_overview"]:
        to = data["tech_overview"]
        tech_overview = TechOverview(
            description=to.get("description", ""),
            technologies=to.get("technologies", [])
        )

    # Parse topics
    topics = []
    for t in data.get("topics", []):
        topics.append(ITTopicSummary(
            topic=t.get("topic", ""),
            summary=t.get("summary", ""),
            speakers=t.get("speakers", []),
            status=t.get("status", "in_progress"),
            issue_refs=t.get("issue_refs", [])
        ))

    # Parse blockers
    blockers = []
    for b in data.get("blockers", []):
        blockers.append(Blocker(
            issue=b.get("issue", ""),
            owner=b.get("owner", ""),
            needs=b.get("needs", "")
        ))

    # Parse action items
    action_items = []
    for a in data.get("action_items", []):
        if isinstance(a, str):
            # Fallback for simple string format
            action_items.append(ActionItem(task=a))
        else:
            action_items.append(ActionItem(
                task=a.get("task", ""),
                assignee=a.get("assignee", ""),
                deadline=a.get("deadline"),
                priority=a.get("priority", "medium")
            ))

    return ITSummaryResult(
        overall_summary=data.get("overall_summary", ""),
        tech_overview=tech_overview,
        topics=topics,
        blockers=blockers,
        action_items=action_items,
        decisions=data.get("decisions", []),
        next_steps=data.get("next_steps", [])
    )
