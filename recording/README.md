# Recording Folder

이 폴더는 참가자의 발음 평가를 위한 녹음 파일을 저장하는 곳입니다.

## 사용 방법

### 1. 녹음하기

Windows 녹음기 또는 다른 녹음 프로그램을 사용하여 다음 문장을 녹음하세요:

**"The quick brown fox jumps over the lazy dog."**

### 2. 파일 저장

- 녹음한 파일을 **WAV 형식**으로 이 폴더(`recording/`)에 저장하세요
- 파일명은 자유롭게 지정할 수 있습니다 (예: `participant1.wav`, `my_recording.wav`)

### 3. 발음 평가 실행

프로젝트 루트 디렉토리에서 다음 명령어를 실행하세요:

```bash
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint <모델_체크포인트_경로>
```

#### 옵션 설명:
- `--mode participant`: 참가자 발음 평가 모드
- `--checkpoint`: 학습된 모델 체크포인트 파일 경로 (필수)
- `--phoneme_map`: phoneme_to_id.json 파일 경로 (기본값: `data/phoneme_to_id.json`)
- `--audio`: 특정 파일명을 지정 (선택사항, 미지정 시 폴더의 첫 번째 .wav 파일 사용)

#### 예시:

```bash
# 기본 사용 (폴더의 첫 번째 .wav 파일 자동 선택)
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt

# 특정 파일 지정
python -m LopeScript.Demo.demo \
    --mode participant \
    --checkpoint checkpoints/best_model.pt \
    --audio participant1.wav
```

### 4. 결과 확인

프로그램이 실행되면 다음 정보가 출력됩니다:
- 📝 참조 텍스트 (Reference Text)
- 🎤 사용된 오디오 파일
- 📊 표준 음소 (Canonical Phonemes)
- 🔍 예측된 음소 (Predicted Phonemes)
- 📈 음소 오류율 (PER - Phoneme Error Rate)
- 💬 평가 결과 (Excellent/Good/Fair/Needs improvement)

## 주의사항

- 녹음 파일은 반드시 **WAV 형식**이어야 합니다
- 녹음 시 배경 소음을 최소화하세요
- 명확한 발음으로 녹음하세요
- 여러 참가자의 녹음을 비교하려면 각각 다른 파일명으로 저장하세요

## 평가 기준

- **PER < 0.1 (10%)**: Excellent! 🌟🌟🌟
- **PER < 0.2 (20%)**: Good! 🌟🌟
- **PER < 0.3 (30%)**: Fair 🌟
- **PER ≥ 0.3 (30%)**: Needs improvement

낮은 PER 값일수록 발음이 표준 음소에 가깝다는 의미입니다.
