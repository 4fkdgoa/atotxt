"""Tests for the summarizer service."""

from app.models.schemas import Transcript, Utterance
from app.services.summarizer import _parse_summary


def test_parse_summary_valid_json():
    raw = """{
        "overall_summary": "회의 요약입니다.",
        "topics": [
            {
                "topic": "매출 현황",
                "summary": "3분기 매출이 증가했습니다.",
                "speakers": ["SPEAKER_00"]
            }
        ],
        "action_items": ["보고서 작성"]
    }"""

    result = _parse_summary(raw)
    assert result.overall_summary == "회의 요약입니다."
    assert len(result.topics) == 1
    assert result.topics[0].topic == "매출 현황"
    assert result.action_items == ["보고서 작성"]


def test_parse_summary_with_markdown_fences():
    raw = """```json
{
    "overall_summary": "요약 테스트",
    "topics": [],
    "action_items": []
}
```"""

    result = _parse_summary(raw)
    assert result.overall_summary == "요약 테스트"


def test_parse_summary_with_extra_text():
    raw = """Here is the analysis:
{
    "overall_summary": "분석 결과",
    "topics": [{"topic": "test", "summary": "test summary", "speakers": []}],
    "action_items": []
}
That's the summary."""

    result = _parse_summary(raw)
    assert result.overall_summary == "분석 결과"
    assert len(result.topics) == 1


def test_parse_summary_invalid_json():
    raw = "This is not JSON at all."
    result = _parse_summary(raw)
    assert "This is not JSON" in result.overall_summary


def test_transcript_to_dialogue_text():
    transcript = Transcript(
        language="ko",
        utterances=[
            Utterance(speaker="SPEAKER_00", text="안녕하세요"),
            Utterance(speaker="SPEAKER_01", text="네 반갑습니다"),
        ],
    )
    text = transcript.to_dialogue_text()
    assert "SPEAKER_00: 안녕하세요" in text
    assert "SPEAKER_01: 네 반갑습니다" in text
