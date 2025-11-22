"""
Minimal configuration for perceived-only pronunciation assessment model.

This config is optimized for:
  - single-task perceived phoneme CTC training
  - Wav2Vec2 encoder + simple feature encoder + PhonemeHead
  - the current Model, Trainer, train.py, evaluation.py pipeline
"""

from dataclasses import dataclass


@dataclass
class Config:
    # -----------------------------
    # Model settings
    # -----------------------------
    # HuggingFace Wav2Vec2 모델 이름 (encoder에서 사용)
    pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53"

    # 오디오 관련 기본 세팅
    sampling_rate: int = 16000
    max_length: int = 160000   # 대략 10초 (원하면 나중에 조정)

    # 출력 클래스 수 (blank 포함)
    num_phonemes: int = 42
    dropout: float = 0.1

    # -----------------------------
    # Training hyperparameters
    # -----------------------------
    # Trainer에서 쓰는 두 개의 learning rate
    main_lr: float = 1e-4          # head / feature encoder
    wav2vec_lr: float = 1e-5       # Wav2Vec2 encoder (더 작게)

    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 20

    # gradient accumulation (Trainer에서 사용)
    gradient_accumulation: int = 1

    # -----------------------------
    # Loss settings (Focal / CTC)
    # -----------------------------
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # CTC에서 blank 토큰 인덱스
    blank_id: int = 0

    # -----------------------------
    # Data augmentation / mode
    # -----------------------------
    # Trainer.train_epoch 에서 specaugment 켤지 여부
    wav2vec2_specaug: bool = True

    # training_mode는 지금은 perceived-only지만
    # Trainer 코드에서 참조하고 있으니 필드만 유지
    training_mode: str = "perceived"

    # -----------------------------
    # Device / 기타
    # -----------------------------
    device: str = "cuda"
    seed: int = 42


def get_config(path: str | None = None) -> Config:
    """
    단순 버전 config 로더.

    현재는 path를 사용하지 않고 항상 기본 Config()를 반환하지만,
    나중에 YAML/JSON에서 읽어오고 싶으면 여기서 확장하면 된다.
    """
    return Config()
