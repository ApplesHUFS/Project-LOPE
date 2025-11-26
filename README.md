# Pronunciation Assessment

Wav2Vec2 기반 발음 평가 모델.

## 설치

```bash
# 가상환경 생성 및 활성화
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

### 1. 데이터셋 다운로드

```bash
bash download_dataset.sh
```

### 2. 전처리

전처리와 데이터 분할을 한 번에 실행합니다.

```bash
python main.py --mode preprocess \
  --data_root data/l2arctic \
  --output data/preprocessed_perceived.json
```

출력: `data/train.json`, `data/dev.json`, `data/test.json`

### 3. 학습

```bash
python main.py --mode train \
  --train_data data/train.json \
  --val_data data/dev.json \
  --save_dir experiments
```

### 4. 평가

```bash
python main.py --mode eval \
  --test_data data/test.json \
  --checkpoint experiments/best_model.pt
```

### 5. 데모

**Test dataset 샘플 사용:**

```bash
python -m LopeScript.Demo.demo \
  --mode test_dataset \
  --checkpoint experiments/best_model.pt \
  --test_data data/test.json \
  --sample_index 0
```

**사용자 오디오 파일 사용:**

```bash
python -m LopeScript.Demo.demo \
  --mode custom \
  --checkpoint experiments/best_model.pt \
  --audio path/to/audio.wav \
  --text "Your transcript here"
```

## License

MIT
