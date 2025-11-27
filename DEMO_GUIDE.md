# 발음 평가 데모 가이드

이 문서는 "The quick brown fox jumps over the lazy dog." 문장을 사용한 참가자 발음 평가 데모의 전체 사용 방법을 설명합니다.

## 📋 목차
1. [준비 사항](#준비-사항)
2. [빠른 시작](#빠른-시작)
3. [상세 사용법](#상세-사용법)
4. [출력 결과 이해하기](#출력-결과-이해하기)
5. [문제 해결](#문제-해결)

---

## 🎯 준비 사항

### 필수 요구사항
- ✅ 학습된 모델 체크포인트 파일 (.pt)
- ✅ `data/phoneme_to_id.json` 파일
- ✅ Python 환경 및 필요한 패키지 설치
- ✅ 녹음 장비 (마이크)

### 파일 구조
```
pronunciation_assessment/
├── run_demo.py              # 간편 실행 스크립트
├── recording/               # 녹음 파일 저장 폴더
│   └── README.md           # 녹음 가이드
├── checkpoints/            # 모델 체크포인트 (생성 필요)
│   └── best_model.pt
├── data/
│   └── phoneme_to_id.json
└── LopeScript/
    └── Demo/
        └── demo.py         # 데모 코드
```

---

## 🚀 빠른 시작

### 방법 1: 간편 실행 스크립트 사용 (추천)

<<<<<<< HEAD
1. **모델 체크포인트 준비**
   ```bash
   # checkpoints 폴더 생성
   mkdir -p checkpoints
   # 학습된 모델을 checkpoints/best_model.pt로 복사
   ```

2. **문장 녹음**
=======

Step 1: 문장 녹음
>>>>>>> 5a71289 (final update)
   - Windows 녹음기를 실행
   - 다음 문장을 명확하게 녹음: **"The quick brown fox jumps over the lazy dog."**
   - `recording/` 폴더에 WAV 형식으로 저장 (예: `participant1.wav`)

<<<<<<< HEAD
3. **데모 실행**
   ```bash
   python run_demo.py
   ```

### 방법 2: 직접 명령어 사용

```bash
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt \
    --phoneme_map data/phoneme_to_id.json
```
=======

   ```

Step 2: 직접 명령어 사용

python -m LopeScript.Demo.demo --mode participant --checkpoint experiments/best_model.pt --audio (recording.wav 이름적기)
>>>>>>> 5a71289 (final update)

---

## 📖 상세 사용법

### 1단계: 녹음 준비

#### Windows 녹음기 사용법
1. 시작 메뉴에서 "음성 녹음기" 또는 "Voice Recorder" 검색
2. 녹음 버튼(●) 클릭
3. 다음 문장을 명확하게 발음:
   ```
   The quick brown fox jumps over the lazy dog.
   ```
4. 정지 버튼(■) 클릭
5. 녹음 파일을 WAV 형식으로 변환하여 저장

#### 녹음 팁
- 🎤 배경 소음이 적은 조용한 환경에서 녹음
- 🗣️ 자연스럽고 명확한 발음으로 녹음
- 📊 표준 발음에 가깝게 발음 (정확도가 높을수록 PER이 낮음)
- ⏱️ 너무 빠르거나 느리지 않게 적당한 속도로

### 2단계: 파일 배치

녹음한 WAV 파일을 `recording/` 폴더에 저장:

```bash
recording/
├── participant1.wav
├── participant2.wav
└── test_recording.wav
```

### 3단계: 실행

#### 옵션 A: run_demo.py 사용

```bash
# 기본 실행 (첫 번째 WAV 파일 자동 선택)
python run_demo.py
```

`run_demo.py` 파일 내에서 체크포인트 경로 수정:
```python
CHECKPOINT_PATH = "checkpoints/best_model.pt"  # 실제 경로로 수정
```

#### 옵션 B: 직접 명령어 사용

```bash
# 기본 실행
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt

# 특정 파일 지정
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt \
    --audio participant1.wav

# 커스텀 phoneme map 사용
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt \
    --phoneme_map data/phoneme_to_id.json \
    --audio my_recording.wav
```

### 명령어 옵션 설명

| 옵션 | 필수/선택 | 설명 | 기본값 |
|------|----------|------|--------|
| `--mode` | 필수 | 데모 모드 (`participant`) | - |
| `--checkpoint` | 필수 | 모델 체크포인트 파일 경로 | - |
| `--phoneme_map` | 선택 | phoneme_to_id.json 경로 | `data/phoneme_to_id.json` |
| `--audio` | 선택 | recording/ 내 파일명 | 첫 번째 .wav 파일 |

---

## 📊 출력 결과 이해하기

### 출력 예시

```
============================================================
       PRONUNCIATION ASSESSMENT DEMO
============================================================

📝 Reference Text:
   The quick brown fox jumps over the lazy dog.

🎤 Audio File: recording/participant1.wav
🤖 Model Checkpoint: checkpoints/best_model.pt

────────────────────────────────────────────────────────────
📊 Canonical Phonemes (35):
   DH AH0 K W IH1 K B R AW1 N F AA1 K S JH AH0 M P S OW1 V ER0 DH AH0 L EY1 Z IY0 D AO1 G

🔍 Predicted Phonemes (33):
   DH AH0 K W IH1 K B R AW1 N F AA1 K S JH AH0 M P S OW1 V ER0 DH AH0 L EY1 Z IY0 D AO1

────────────────────────────────────────────────────────────
📈 Phoneme Error Rate (PER): 0.0857 (8.57%)
💬 Assessment: Excellent! 🌟🌟🌟
============================================================
```

### 결과 항목 설명

1. **📝 Reference Text**: 평가 기준 문장
2. **🎤 Audio File**: 분석된 녹음 파일
3. **📊 Canonical Phonemes**: 표준 발음의 음소 시퀀스
4. **🔍 Predicted Phonemes**: 모델이 예측한 실제 발음의 음소 시퀀스
5. **📈 PER (Phoneme Error Rate)**: 음소 오류율
   - 낮을수록 좋음 (0에 가까울수록 완벽)
   - 삽입/삭제/대체 오류를 기반으로 계산

### 평가 등급

| PER 범위 | 등급 | 의미 |
|----------|------|------|
| < 0.1 (10%) | Excellent! 🌟🌟🌟 | 거의 완벽한 발음 |
| 0.1 ~ 0.2 (10-20%) | Good! 🌟🌟 | 좋은 발음 |
| 0.2 ~ 0.3 (20-30%) | Fair 🌟 | 보통 발음 |
| ≥ 0.3 (30%+) | Needs improvement | 개선 필요 |

---

## 🔧 문제 해결

### 문제 1: 체크포인트 파일을 찾을 수 없음

**증상:**
```
FileNotFoundError: Checkpoint not found: checkpoints/best_model.pt
```

**해결방법:**
1. 체크포인트 파일이 올바른 위치에 있는지 확인
2. `run_demo.py`의 `CHECKPOINT_PATH` 변수를 실제 경로로 수정
3. 또는 명령어 실행 시 `--checkpoint` 옵션에 정확한 경로 지정

### 문제 2: 녹음 파일이 없음

**증상:**
```
Error: No WAV files found in 'recording' directory
```

**해결방법:**
1. `recording/` 폴더가 존재하는지 확인
2. 녹음 파일이 WAV 형식인지 확인 (MP3, M4A 등은 지원하지 않음)
3. 파일 확장자가 `.wav`인지 확인

### 문제 3: phoneme_to_id.json 파일이 없음

**증상:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/phoneme_to_id.json'
```

**해결방법:**
1. 데이터 전처리를 먼저 실행하여 파일 생성:
   ```bash
   python preprocess.py
   ```
2. 또는 다른 위치의 파일을 `--phoneme_map` 옵션으로 지정

### 문제 4: 모듈을 찾을 수 없음

**증상:**
```
ModuleNotFoundError: No module named 'LopeScript'
```

**해결방법:**
1. 프로젝트 루트 디렉토리에서 실행하고 있는지 확인
2. 필요한 패키지가 설치되어 있는지 확인:
   ```bash
   pip install -r requirements.txt
   ```

### 문제 5: CUDA 관련 오류

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결방법:**
1. CPU 모드로 실행 (config.py에서 device 설정 변경)
2. 또는 더 작은 배치 크기 사용

---

## 💡 추가 팁

### 여러 참가자 평가하기

```bash
# 각 참가자의 녹음을 다른 이름으로 저장
recording/
├── participant_01.wav
├── participant_02.wav
├── participant_03.wav
└── ...

# 각 파일을 순차적으로 평가
python -m LopeScript.Demo.demo --mode participant --checkpoint checkpoints/best_model.pt --audio participant_01.wav
python -m LopeScript.Demo.demo --mode participant --checkpoint checkpoints/best_model.pt --audio participant_02.wav
# ...
```

### 결과 저장하기

```bash
# 출력을 파일로 저장
python run_demo.py > results.txt 2>&1

# 또는 특정 참가자 결과 저장
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt \
    --audio participant1.wav > participant1_result.txt
```

### 배치 평가 스크립트

여러 파일을 자동으로 평가하려면:

```bash
# recording 폴더의 모든 WAV 파일 평가
for file in recording/*.wav; do
    filename=$(basename "$file")
    echo "Evaluating: $filename"
    python -m LopeScript.Demo.demo \
        --mode participant \
        --checkpoint checkpoints/best_model.pt \
        --audio "$filename" > "results/${filename%.wav}_result.txt"
done
```

---

## 📞 추가 지원

더 자세한 정보는 다음 문서를 참조하세요:
- `README.md`: 프로젝트 전체 개요
- `recording/README.md`: 녹음 가이드
- `LopeScript/Demo/demo.py`: 데모 코드 상세 구현

---

**마지막 업데이트**: 2025-11-26
