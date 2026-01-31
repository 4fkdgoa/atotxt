# 파인튜닝 가이드 (초등학생도 가능)

## 파인튜닝이 뭐야?

```
AI한테 "이렇게 대답해!" 라고 예시를 많이 보여주는 거야.
예시를 많이 보여줄수록 AI가 더 똑똑해져!
```

---

## 1단계: 데이터 모으기

회의 녹음할 때마다 자동으로 데이터가 쌓여요.

```
회의 녹음 → 전사 → 요약 → 저장!
         ↓
    [transcript + summary]
         ↓
    파인튜닝 데이터 완성!
```

### 데이터 형식 예시

```json
{
  "input": "SPEAKER_00: 오늘 스프린트 리뷰 시작합니다\nSPEAKER_01: 저는 로그인 API 버그 수정했어요",
  "output": {
    "overall_summary": "스프린트 리뷰 - 로그인 버그 수정 완료",
    "topics": [{"topic": "로그인 API", "status": "done"}],
    "action_items": []
  }
}
```

### 얼마나 모아야 해?

| 데이터 수 | 효과 |
|----------|------|
| 50개 | 시작 가능 (기본) |
| 100개 | 괜찮음 |
| 500개 | 좋음 |
| 1000개+ | 아주 좋음 |

---

## 2단계: 데이터 저장 설정

`.env` 파일에 추가:

```bash
# 파인튜닝 데이터 저장
SAVE_TRAINING_DATA=true
TRAINING_DATA_DIR=./training_data
```

그러면 이렇게 저장돼요:

```
training_data/
├── general/
│   ├── 2024-01-15_meeting_001.json
│   └── 2024-01-16_meeting_002.json
└── it_standup/
    ├── 2024-01-15_standup_001.json
    └── 2024-01-16_standup_002.json
```

---

## 3단계: 파인튜닝 실행

### 방법 A: Unsloth (가장 쉬움, 무료)

```bash
# 1. 설치
pip install unsloth

# 2. 파인튜닝 실행
python scripts/finetune.py \
  --data ./training_data/it_standup \
  --model qwen2.5:7b \
  --output ./my_model
```

### 방법 B: Google Colab (컴퓨터 느려도 OK)

1. https://colab.research.google.com 접속
2. 아래 코드 복사해서 실행

```python
# 1단계: 설치
!pip install unsloth transformers datasets

# 2단계: 데이터 업로드
from google.colab import files
files.upload()  # training_data.json 업로드

# 3단계: 파인튜닝
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B",  # 기본 모델
    max_seq_length=2048,
)

# LoRA 설정 (가벼운 튜닝)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 작을수록 가벼움
    target_modules=["q_proj", "v_proj"],
)

# 학습!
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,
)
trainer.train()

# 4단계: 저장
model.save_pretrained("my_finetuned_model")
```

### 방법 C: Ollama에 직접 등록

```bash
# 1. GGUF로 변환
python -m llama.cpp.convert my_model --outfile my_model.gguf

# 2. Modelfile 작성
cat > Modelfile << 'EOF'
FROM ./my_model.gguf
SYSTEM "당신은 IT 회의록 분석 전문가입니다."
EOF

# 3. Ollama에 등록
ollama create my-meeting-ai -f Modelfile

# 4. 테스트
ollama run my-meeting-ai "회의 요약해줘"
```

---

## 4단계: 파인튜닝된 모델 사용

`.env` 수정:

```bash
# 기존
OLLAMA_MODEL=qwen2.5:14b

# 변경 (내가 만든 모델로!)
OLLAMA_MODEL=my-meeting-ai
```

끝!

---

## 요약: 전체 흐름

```
┌─────────────────────────────────────────────────────┐
│  1. 회의 녹음                                        │
│     ↓                                               │
│  2. atotxt로 전사 + 요약 (meeting_type 선택)         │
│     ↓                                               │
│  3. 데이터 자동 저장 (training_data/)                │
│     ↓                                               │
│  4. 50개 이상 모이면 파인튜닝                        │
│     ↓                                               │
│  5. 새 모델 Ollama에 등록                           │
│     ↓                                               │
│  6. .env에서 모델 변경                              │
│     ↓                                               │
│  7. 더 똑똑해진 AI 사용!                            │
└─────────────────────────────────────────────────────┘
```

---

## 자주 묻는 질문

### Q: GPU 없어도 돼?
A: Google Colab 쓰면 무료 GPU 사용 가능!

### Q: 데이터 형식 틀리면?
A: 자동으로 맞춰주는 스크립트 제공 (scripts/prepare_data.py)

### Q: 얼마나 걸려?
A: 데이터 100개 기준 약 30분~1시간

### Q: 파인튜닝 실패하면?
A: 기존 qwen2.5 모델 그대로 사용하면 됨 (손해 없음!)

---

## 다음 단계

데이터 저장 기능 켜고 싶으면:

```bash
# .env에 추가
SAVE_TRAINING_DATA=true
```

50개 모이면 알려드릴게요!
