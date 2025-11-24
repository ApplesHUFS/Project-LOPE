# Demo/demo.py

"""
Demo: single-sentence pronunciation assessment

Target sentence (canonical):
    "The quick brown fox jumps over the lazy dog."

Pipeline:
  1) Load trained model checkpoint
  2) Load a single WAV file
  3) Run model -> CTC greedy decoding -> predicted phoneme IDs
  4) Compare with canonical phoneme IDs
  5) Print PER and both sequences
"""

import argparse
import os

import torch
import torchaudio

from config import get_config
from Model.model import Model
from Evaluation.eval import greedy_ctc_decode
from Evaluation.metric import utterance_per
from utils.audio import create_attention_mask, compute_output_lengths


# 평가할 문장 (고정)
TARGET_SENTENCE = "The quick brown fox jumps over the lazy dog."

CANONICAL_PHONEME_IDS = [
    11, 4,        # dh ah  (The)
    21, 37, 18, 21,  # k w ih k   (quick)
    8, 29, 6, 24,    # b r aw n   (brown)
    15, 2, 21, 30,   # f aa k s   (fox)
    20, 4, 23, 28, 30,  # jh ah m p s (jumps)
    26, 36, 13,   # ow v er    (over)
    11, 4,        # dh ah      (the)
    22, 14, 39, 19,  # l ey z iy (lazy)
    10, 5, 16     # d ao g     (dog)
]


def load_single_waveform(path: str, config):
    """하나의 wav 파일을 로드하고 padding/cut 해서 [1, T] 텐서로 반환."""
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono

    if sr != config.sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, config.sampling_rate)

    wav = wav[0]  # [T]
    length = wav.size(0)

    # max_length 기준으로 자르거나 padding
    if length > config.max_length:
        wav = wav[:config.max_length]
        length = config.max_length
    else:
        pad_len = config.max_length - length
        if pad_len > 0:
            wav = torch.nn.functional.pad(wav, (0, pad_len))

    wav = wav.unsqueeze(0)  # [1, T]
    audio_lengths = torch.tensor([length], dtype=torch.long)
    return wav, audio_lengths


def run_demo(audio_path: str, checkpoint: str):
    """실제 데모 한 번 수행.

    Args:
        audio_path: 입력 wav 파일 경로
        checkpoint: 학습된 모델 체크포인트 (.pt)
    """
    config = get_config()
    device = config.device if torch.cuda.is_available() else "cpu"

    # 1) 모델 생성 & 체크포인트 로드
    model = Model(
        pretrained_model_name=config.pretrained_model_name,
        num_phonemes=config.num_phonemes,
        dropout=config.dropout,
    )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    # train.py에서 {"model_state": ...} 형식으로 저장했다면:
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 2) 오디오 로드
    waveforms, audio_lengths = load_single_waveform(audio_path, config)
    waveforms = waveforms.to(device)
    audio_lengths = audio_lengths.to(device)

    # 3) attention mask / input_lengths 계산
    input_lengths = compute_output_lengths(model, audio_lengths)
    normalized_lengths = audio_lengths.float() / waveforms.shape[1]
    attention_mask = create_attention_mask(waveforms, normalized_lengths)

    # 4) 모델 forward + CTC greedy decode
    with torch.no_grad():
        logits = model(waveforms, attention_mask)  # [1, T_enc, C]
        hyps = greedy_ctc_decode(logits, blank_id=config.blank_id)
        predicted_ids = hyps[0]  # length 가변 list[int]

    # 5) canonical과 PER 계산
    per_value = utterance_per(CANONICAL_PHONEME_IDS, predicted_ids)

    # 6) 결과 출력
    print("\n=== DEMO RESULT ===")
    print(f"Target sentence: {TARGET_SENTENCE}")
    print(f"Audio file      : {audio_path}")
    print(f"Checkpoint      : {checkpoint}")
    print(f"\nCanonical IDs ({len(CANONICAL_PHONEME_IDS)}):")
    print(CANONICAL_PHONEME_IDS)
    print(f"\nPredicted IDs ({len(predicted_ids)}):")
    print(predicted_ids)
    print(f"\nPER (canonical vs predicted): {per_value:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: single-sentence pronunciation assessment"
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="입력 wav 파일 경로",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="학습된 모델 체크포인트 경로 (.pt)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_demo(
        audio_path=args.audio,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
