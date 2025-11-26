"""Configuration for pronunciation assessment model."""

from dataclasses import dataclass


@dataclass
class Config:
  pretrained_model_name: str = "facebook/wav2vec2-base"

  sampling_rate: int = 16000
  max_length: int = 160000

  num_phonemes: int = 42
  dropout: float = 0.1

  main_lr: float = 1e-4
  wav2vec_lr: float = 1e-5

  batch_size: int = 32
  eval_batch_size: int = 32
  num_workers: int = 4
  num_epochs: int = 30

  gradient_accumulation: int = 2

  focal_alpha: float = 0.25
  focal_gamma: float = 2.0

  blank_id: int = 0

  wav2vec2_specaug: bool = True

  training_mode: str = "perceived"

  device: str = "cuda"
  seed: int = 42


def get_config(path: str | None = None) -> Config:
  """Returns default configuration."""
  return Config()
