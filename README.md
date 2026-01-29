# atotxt - Audio to Text

음성 파일을 텍스트로 변환하고, 화자별 분리(다이어라이제이션) + AI 요약/주제 분류까지 자동화하는 서비스.

## 주요 기능

- **음성 전사 (STT)**: WhisperX 기반, 한국어 최적화
- **화자 분리**: 발화자 자동 구분 (SPEAKER_00, SPEAKER_01, ...)
- **AI 요약**: Ollama LLM으로 주제별 요약 및 액션 아이템 추출
- **GPU 자동 감지**: VRAM 크기에 따라 최적 모델 자동 선택
- **REST API**: FastAPI 기반 업로드/조회 API
- **비동기 처리**: 대용량 파일 비동기 처리 지원

## 라이선스 호환성

| 컴포넌트 | 라이선스 | 상용 사용 |
|----------|---------|----------|
| WhisperX | BSD-4 | OK |
| Ollama | MIT | OK |
| FastAPI | MIT | OK |

## 프로젝트 구조

```
atotxt/
├── main.py                          # 서버 진입점
├── requirements.txt                 # Python 의존성
├── docker-compose.yml               # Docker 구성
├── Dockerfile                       # CPU 빌드
├── Dockerfile.gpu                   # GPU 빌드 (CUDA)
├── .env.example                     # 환경 변수 템플릿
├── app/
│   ├── api/
│   │   └── routes.py                # API 엔드포인트
│   ├── core/
│   │   └── config.py                # 설정 관리
│   ├── models/
│   │   └── schemas.py               # Pydantic 스키마
│   └── services/
│       ├── gpu_detector.py          # GPU 감지 + 모델 선택
│       ├── transcriber.py           # WhisperX 전사
│       ├── summarizer.py            # Ollama 요약
│       └── queue_manager.py         # 작업 큐 관리
├── scripts/
│   ├── install_windows.ps1          # Windows 설치 스크립트
│   └── install_linux.sh             # Linux 설치 스크립트
└── tests/
    ├── test_api.py
    ├── test_summarizer.py
    └── test_gpu_detector.py
```

## 빠른 시작

### 1. 설치

**Windows Server:**

```powershell
.\scripts\install_windows.ps1
```

**Linux:**

```bash
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
```

**수동 설치:**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Ollama 설정

```bash
# Ollama 설치 (https://ollama.com)
ollama pull qwen2.5:14b   # GPU 16GB+
# 또는
ollama pull qwen2.5:7b    # GPU 8GB 이하 / CPU
```

### 3. 서버 실행

```bash
python main.py
```

서버가 `http://localhost:8000`에서 시작됩니다.

### 4. Docker로 실행

```bash
# CPU
docker compose up

# GPU
docker compose --profile gpu up
```

## API 사용법

### 비동기 업로드 (대용량 파일 권장)

```bash
# 업로드 → 즉시 task_id 반환
curl -X POST http://localhost:8000/upload \
  -F "file=@meeting.wav"

# 결과 조회
curl http://localhost:8000/task/{task_id}
```

### 동기 전사 (소용량 파일)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@short_clip.wav"
```

### 화자 수 지정

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@meeting.wav" \
  -F "num_speakers=3"
```

### Python 클라이언트

```python
import requests

# 업로드
files = {"file": open("meeting.wav", "rb")}
r = requests.post("http://localhost:8000/upload", files=files)
task_id = r.json()["task_id"]

# 결과 확인 (폴링)
import time
while True:
    r = requests.get(f"http://localhost:8000/task/{task_id}")
    data = r.json()
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(2)

print(data)
```

## API 응답 예시

```json
{
  "task_id": "a1b2c3d4",
  "status": "completed",
  "transcript": {
    "language": "ko",
    "utterances": [
      {
        "speaker": "SPEAKER_00",
        "text": "오늘 회의 시작하겠습니다.",
        "start": 0.5,
        "end": 2.1
      },
      {
        "speaker": "SPEAKER_01",
        "text": "이번 분기 매출 보고 드리겠습니다.",
        "start": 2.5,
        "end": 5.3
      }
    ]
  },
  "summary": {
    "overall_summary": "매출 현황 보고 및 마케팅 전략 논의",
    "topics": [
      {
        "topic": "매출 현황",
        "summary": "3분기 매출은 전년 대비 12% 증가",
        "speakers": ["SPEAKER_00", "SPEAKER_01"]
      },
      {
        "topic": "마케팅 전략",
        "summary": "온라인 광고 예산 증액 필요성 논의",
        "speakers": ["SPEAKER_01"]
      }
    ],
    "action_items": [
      "마케팅 예산안 다음 주까지 제출",
      "매출 보고서 전체 팀 공유"
    ]
  }
}
```

## GPU별 권장 설정

| GPU | VRAM | Whisper 모델 | LLM 모델 | 용도 |
|-----|------|-------------|----------|------|
| CPU only | - | base | qwen2.5:7b | 개발/테스트 |
| GTX 1660 | 6GB | small (int8) | llama3:8b | 배치 처리 |
| RTX 3060 | 12GB | medium (int8) | qwen2.5:7b | 일반 서비스 |
| RTX 5070 Ti | 16GB | large-v3 (fp16) | qwen2.5:14b | 상용 서비스 |

`WHISPER_MODEL=auto`로 설정하면 GPU VRAM에 따라 자동 선택됩니다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `HOST` | 0.0.0.0 | 서버 바인드 주소 |
| `PORT` | 8000 | 서버 포트 |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API 주소 |
| `OLLAMA_MODEL` | qwen2.5:14b | 요약용 LLM 모델 |
| `WHISPER_MODEL` | auto | Whisper 모델 (auto=자동 선택) |
| `WHISPER_LANGUAGE` | ko | 음성 언어 |
| `WHISPER_DEVICE` | auto | cpu 또는 cuda |
| `HF_TOKEN` | | HuggingFace 토큰 (화자 분리용) |
| `REDIS_URL` | redis://localhost:6379/0 | Redis 주소 |
| `MAX_UPLOAD_SIZE_MB` | 500 | 최대 업로드 크기 |

## 화자 분리 설정

화자 분리(Diarization)를 사용하려면 HuggingFace 토큰이 필요합니다:

1. https://huggingface.co/settings/tokens 에서 토큰 생성
2. https://huggingface.co/pyannote/speaker-diarization-3.1 에서 사용 동의
3. `.env` 파일에 `HF_TOKEN=hf_your_token` 설정

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/upload` | 오디오 업로드 (비동기) |
| GET | `/task/{task_id}` | 작업 상태/결과 조회 |
| POST | `/transcribe` | 오디오 업로드 (동기) |
| GET | `/health` | 헬스 체크 |
| GET | `/system` | 시스템/GPU 정보 |

## 지원 오디오 포맷

wav, mp3, m4a, flac, ogg, wma, aac, webm, mp4
