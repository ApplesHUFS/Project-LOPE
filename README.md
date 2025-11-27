# Pronunciation Assessment

Wav2Vec2 기반 발음 평가 모델.

## 시스템 아키텍처
LopeScript/
├── DataProc/           # 데이터 전처리 프로세스
├── Demo/            # 데이터별 음소 평가
├── Model/    # 모델 인코더 및 통합 모델
├── Train/         # 모델 학습
└── Utils/            

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

본 프로젝트는 https://creativecommons.org/licenses/by-nc-sa/4.0/ 하에 배포됩니다.

### 라이센스 조건
#### 저작자 표시
적절한 출처를 명시하고, 라이선스 링크를 제공하며, 변경 사항이 있을 경우 이를 표시해야 합니다.
#### 비영리
본 저작물은 비영리 목적으로만 사용할 수 있으며, 상업적 이용은 허용되지 않습니다.
#### 동일조건변경허락
본 저작물을 개작, 변형 또는 가공했을 경우 반드시 원저작물과 동일한 라이센스 조건으로 배포하여야 합니다.

라이센스에 대한 더 자세한 사항은 https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ko에서 확인하세요.

#### 포함된 제3자 저작물
L2-ARCTIC: a non-native English speech corpus

## 프로젝트 데모 사용법
데모 파일 사용법에 대해서는 다음 파일을 참고하세요.
DEMO_GUIDE.md

- 제작자 : Texas A&M University, Iowa State University
- 출처 : https://psi.engr.tamu.edu/l2-arctic-corpus/
- 라이센스 : CC BY-NC 4.0

프로젝트에 사용된 L2 arctic 영어 비모국어 음성 코퍼스는  Texas A&M University와 Iowa State University 연구자들이 공동으로 작업한 데이터로 CC BY-NC 4.0 라이선스에 따라 배포됩니다.
