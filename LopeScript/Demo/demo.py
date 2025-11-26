"""Demo: pronunciation assessment."""

import argparse
import json
import os
import sys
from typing import Optional
from datetime import datetime

import torch
import torchaudio

from ..config import get_config
from ..Model.model import Model
from ..Evaluation.eval import greedy_ctc_decode
from ..Evaluation.metric import utterance_per
from ..Utils.audio import create_attention_mask, compute_output_lengths
from ..Utils.cmu_dict import get_canonical_phoneme_ids


def load_single_waveform(path: str, config):
    """단일 WAV 파일 로드 및 전처리.

    Args:
        path: WAV 파일 경로
        config: 설정 객체

    Returns:
        waveform: [1, max_length] 형태의 텐서
        audio_lengths: 실제 오디오 길이 텐서
    """
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != config.sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, config.sampling_rate)

    wav = wav[0]
    length = wav.size(0)

    if length > config.max_length:
        wav = wav[:config.max_length]
        length = config.max_length
    else:
        pad_len = config.max_length - length
        if pad_len > 0:
            wav = torch.nn.functional.pad(wav, (0, pad_len))

    wav = wav.unsqueeze(0)
    audio_lengths = torch.tensor([length], dtype=torch.long)
    return wav, audio_lengths


def run_demo_on_test_dataset(
    checkpoint: str,
    test_data_path: str,
    phoneme_map_path: str,
    sample_index: Optional[int] = None
):
    """Test dataset에서 샘플을 선택하여 평가.

    Args:
        checkpoint: 모델 체크포인트 경로
        test_data_path: test.json 경로
        phoneme_map_path: phoneme_to_id.json 경로
        sample_index: 평가할 샘플의 인덱스 (None이면 첫 번째)
    """
    config = get_config()
    device = config.device if torch.cuda.is_available() else "cpu"

    with open(phoneme_map_path, 'r') as f:
        phoneme_to_id = json.load(f)

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    test_items = list(test_data.items())

    if sample_index is None:
        sample_index = 0

    if sample_index >= len(test_items):
        print(f"Error: sample_index {sample_index} out of range (max: {len(test_items) - 1})")
        sys.exit(1)

    wav_path, sample_info = test_items[sample_index]
    text = sample_info.get('wrd', '')

    if not text:
        print(f"Error: No transcript found for sample {sample_index}")
        sys.exit(1)

    canonical_phoneme_ids = get_canonical_phoneme_ids(text, phoneme_to_id)

    if not canonical_phoneme_ids:
        print(f"Error: Could not generate canonical phonemes for text: {text}")
        sys.exit(1)

    model = Model(
        pretrained_model_name=config.pretrained_model_name,
        num_phonemes=config.num_phonemes,
        dropout=config.dropout,
    )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    waveforms, audio_lengths = load_single_waveform(wav_path, config)
    waveforms = waveforms.to(device)
    audio_lengths = audio_lengths.to(device)

    input_lengths = compute_output_lengths(model, audio_lengths)
    normalized_lengths = audio_lengths.float() / waveforms.shape[1]
    attention_mask = create_attention_mask(waveforms, normalized_lengths)

    with torch.no_grad():
        outputs = model(waveforms, attention_mask)
        logits = outputs['perceived_logits']
        hyps = greedy_ctc_decode(logits, blank_id=config.blank_id)
        predicted_ids = hyps[0]

    per_value = utterance_per(canonical_phoneme_ids, predicted_ids)

    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    canonical_phonemes = [id_to_phoneme.get(pid, f"<{pid}>") for pid in canonical_phoneme_ids]
    predicted_phonemes = [id_to_phoneme.get(pid, f"<{pid}>") for pid in predicted_ids]

    print("\n=== Demo Result (Test Dataset) ===")
    print(f"Sample index: {sample_index}")
    print(f"Text: {text}")
    print(f"Audio file: {wav_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"\nCanonical phonemes ({len(canonical_phonemes)}):")
    print(f"  {' '.join(canonical_phonemes)}")
    print(f"Canonical IDs:")
    print(f"  {canonical_phoneme_ids}")
    print(f"\nPredicted phonemes ({len(predicted_ids)}):")
    print(f"  {' '.join(predicted_phonemes)}")
    print(f"Predicted IDs:")
    print(f"  {predicted_ids}")
    print(f"\nPER: {per_value:.4f}")


def run_demo_with_custom_audio(
    audio_path: str,
    text: str,
    checkpoint: str,
    phoneme_map_path: str
):
    """사용자가 제공한 오디오 파일과 텍스트로 평가.

    Args:
        audio_path: WAV 파일 경로
        text: 발화할 텍스트 (canonical)
        checkpoint: 모델 체크포인트 경로
        phoneme_map_path: phoneme_to_id.json 경로
    """
    config = get_config()
    device = config.device if torch.cuda.is_available() else "cpu"

    with open(phoneme_map_path, 'r') as f:
        phoneme_to_id = json.load(f)

    canonical_phoneme_ids = get_canonical_phoneme_ids(text, phoneme_to_id)

    if not canonical_phoneme_ids:
        print(f"Error: Could not generate canonical phonemes for text: {text}")
        sys.exit(1)

    model = Model(
        pretrained_model_name=config.pretrained_model_name,
        num_phonemes=config.num_phonemes,
        dropout=config.dropout,
    )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    waveforms, audio_lengths = load_single_waveform(audio_path, config)
    waveforms = waveforms.to(device)
    audio_lengths = audio_lengths.to(device)

    input_lengths = compute_output_lengths(model, audio_lengths)
    normalized_lengths = audio_lengths.float() / waveforms.shape[1]
    attention_mask = create_attention_mask(waveforms, normalized_lengths)

    with torch.no_grad():
        outputs = model(waveforms, attention_mask)
        logits = outputs['perceived_logits']
        hyps = greedy_ctc_decode(logits, blank_id=config.blank_id)
        predicted_ids = hyps[0]

    per_value = utterance_per(canonical_phoneme_ids, predicted_ids)

    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    canonical_phonemes = [id_to_phoneme.get(pid, f"<{pid}>") for pid in canonical_phoneme_ids]
    predicted_phonemes = [id_to_phoneme.get(pid, f"<{pid}>") for pid in predicted_ids]

    print("\n=== Demo Result (Custom Audio) ===")
    print(f"Text: {text}")
    print(f"Audio file: {audio_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"\nCanonical phonemes ({len(canonical_phonemes)}):")
    print(f"  {' '.join(canonical_phonemes)}")
    print(f"Canonical IDs:")
    print(f"  {canonical_phoneme_ids}")
    print(f"\nPredicted phonemes ({len(predicted_ids)}):")
    print(f"  {' '.join(predicted_phonemes)}")
    print(f"Predicted IDs:")
    print(f"  {predicted_ids}")
    print(f"\nPER: {per_value:.4f}")


def run_demo_with_participant(
    checkpoint: str,
    phoneme_map_path: str,
    audio_filename: Optional[str] = None
):
    """참가자 발음 평가 데모 (고정된 문장 사용).

    Args:
        checkpoint: 모델 체크포인트 경로
        phoneme_map_path: phoneme_to_id.json 경로
        audio_filename: recording 폴더 내 오디오 파일명 (None이면 첫 번째 wav 파일)
    """
    DEMO_TEXT = "The quick brown fox jumps over the lazy dog."
    RECORDING_DIR = "recording"

    config = get_config()
    device = config.device if torch.cuda.is_available() else "cpu"

    # recording 폴더에서 오디오 파일 찾기
    if audio_filename is None:
        if not os.path.exists(RECORDING_DIR):
            print(f"Error: Recording directory '{RECORDING_DIR}' does not exist")
            sys.exit(1)
        
        wav_files = [f for f in os.listdir(RECORDING_DIR) if f.endswith('.wav')]
        if not wav_files:
            print(f"Error: No WAV files found in '{RECORDING_DIR}' directory")
            print(f"Please record '{DEMO_TEXT}' and save it as a WAV file in the '{RECORDING_DIR}' folder")
            sys.exit(1)
        
        audio_filename = wav_files[0]
        print(f"Using audio file: {audio_filename}")
    
    audio_path = os.path.join(RECORDING_DIR, audio_filename)
    
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    with open(phoneme_map_path, 'r') as f:
        phoneme_to_id = json.load(f)

    canonical_phoneme_ids = get_canonical_phoneme_ids(DEMO_TEXT, phoneme_to_id)

    if not canonical_phoneme_ids:
        print(f"Error: Could not generate canonical phonemes for text: {DEMO_TEXT}")
        sys.exit(1)

    model = Model(
        pretrained_model_name=config.pretrained_model_name,
        num_phonemes=config.num_phonemes,
        dropout=config.dropout,
    )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    waveforms, audio_lengths = load_single_waveform(audio_path, config)
    waveforms = waveforms.to(device)
    audio_lengths = audio_lengths.to(device)

    input_lengths = compute_output_lengths(model, audio_lengths)
    normalized_lengths = audio_lengths.float() / waveforms.shape[1]
    attention_mask = create_attention_mask(waveforms, normalized_lengths)

    with torch.no_grad():
        outputs = model(waveforms, attention_mask)
        logits = outputs['perceived_logits']
        hyps = greedy_ctc_decode(logits, blank_id=config.blank_id)
        predicted_ids = hyps[0]

    per_value = utterance_per(canonical_phoneme_ids, predicted_ids)

    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    canonical_phonemes = [id_to_phoneme.get(pid, f"<{pid}>") for pid in canonical_phoneme_ids]
    predicted_phonemes = [id_to_phoneme.get(pid, f"<{pid}>") for pid in predicted_ids]

    print("\n" + "="*60)
    print("       PRONUNCIATION ASSESSMENT DEMO")
    print("="*60)
    print(f"\n📝 Reference Text:")
    print(f"   {DEMO_TEXT}")
    print(f"\n🎤 Audio File: {audio_path}")
    print(f"🤖 Model Checkpoint: {checkpoint}")
    print(f"\n{'─'*60}")
    print(f"📊 Canonical Phonemes ({len(canonical_phonemes)}):")
    print(f"   {' '.join(canonical_phonemes)}")
    print(f"\n🔍 Predicted Phonemes ({len(predicted_ids)}):")
    print(f"   {' '.join(predicted_phonemes)}")
    print(f"\n{'─'*60}")
    print(f"📈 Phoneme Error Rate (PER): {per_value:.4f} ({per_value*100:.2f}%)")
    
    # 간단한 평가 메시지
    if per_value < 0.1:
        rating = "Excellent! 🌟🌟🌟"
    elif per_value < 0.2:
        rating = "Good! 🌟🌟"
    elif per_value < 0.3:
        rating = "Fair 🌟"
    else:
        rating = "Needs improvement"
    
    print(f"💬 Assessment: {rating}")
    print("="*60 + "\n")
    
    # 결과 저장
    save_result(
        audio_filename=audio_filename,
        audio_path=audio_path,
        text=DEMO_TEXT,
        canonical_phonemes=canonical_phonemes,
        predicted_phonemes=predicted_phonemes,
        canonical_ids=canonical_phoneme_ids,
        predicted_ids=predicted_ids,
        per_value=per_value,
        rating=rating,
        checkpoint=checkpoint
    )


def save_result(
    audio_filename: str,
    audio_path: str,
    text: str,
    canonical_phonemes: list,
    predicted_phonemes: list,
    canonical_ids: list,
    predicted_ids: list,
    per_value: float,
    rating: str,
    checkpoint: str
):
    """평가 결과를 JSON 파일로 저장.
    
    Args:
        audio_filename: 오디오 파일명
        audio_path: 오디오 파일 전체 경로
        text: 평가 대상 텍스트
        canonical_phonemes: 정답 음소 리스트
        predicted_phonemes: 예측 음소 리스트
        canonical_ids: 정답 음소 ID 리스트
        predicted_ids: 예측 음소 ID 리스트
        per_value: Phoneme Error Rate
        rating: 평가 등급
        checkpoint: 사용한 체크포인트 경로
    """
    # results 폴더 생성
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 파일명에서 확장자 제거
    audio_name = os.path.splitext(audio_filename)[0]
    
    # 결과 파일명 생성
    result_filename = f"{audio_name}_{timestamp}.json"
    result_path = os.path.join(results_dir, result_filename)
    
    # 결과 데이터 구성
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "audio_file": audio_path,
        "audio_filename": audio_filename,
        "reference_text": text,
        "checkpoint": checkpoint,
        "canonical": {
            "phonemes": canonical_phonemes,
            "ids": canonical_ids,
            "count": len(canonical_phonemes)
        },
        "predicted": {
            "phonemes": predicted_phonemes,
            "ids": predicted_ids,
            "count": len(predicted_phonemes)
        },
        "evaluation": {
            "per": round(per_value, 4),
            "per_percentage": round(per_value * 100, 2),
            "rating": rating
        }
    }
    
    # JSON 파일로 저장
    with open(result_path, 'w', encoding='utf-8') as f:
        # 먼저 들여쓰기된 JSON으로 변환
        json_str = json.dumps(result_data, indent=2, ensure_ascii=False)
        
        # phonemes와 ids 배열을 한 줄로 변환
        import re
        # "phonemes": [ ... ] 패턴을 찾아서 한 줄로 변경
        json_str = re.sub(r'"phonemes":\s*\[\s*([^\]]+?)\s*\]', 
                         lambda m: '"phonemes": [' + ', '.join(s.strip() for s in m.group(1).split(',')) + ']',
                         json_str, flags=re.DOTALL)
        # "ids": [ ... ] 패턴을 찾아서 한 줄로 변경
        json_str = re.sub(r'"ids":\s*\[\s*([^\]]+?)\s*\]',
                         lambda m: '"ids": [' + ', '.join(s.strip() for s in m.group(1).split(',')) + ']',
                         json_str, flags=re.DOTALL)
        
        f.write(json_str)
    
    print(f"💾 Result saved to: {result_path}")


def main():
    """Demo mode 진입점."""
    parser = argparse.ArgumentParser(description="Demo pronunciation assessment")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["test_dataset", "custom", "participant"],
        help="Demo mode: test_dataset, custom, or participant"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--phoneme_map",
        type=str,
        default="data/phoneme_to_id.json",
        help="Path to phoneme_to_id.json"
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default="data/test.json",
        help="Path to test.json (for test_dataset mode)"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index to use from test dataset"
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Audio file path (for custom mode) or filename in recording/ (for participant mode)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Transcript text (for custom mode)"
    )

    args = parser.parse_args()

    if args.mode == "test_dataset":
        run_demo_on_test_dataset(
            checkpoint=args.checkpoint,
            test_data_path=args.test_data,
            phoneme_map_path=args.phoneme_map,
            sample_index=args.sample_index
        )
    elif args.mode == "custom":
        if not args.audio or not args.text:
            print("Error: --audio and --text are required for custom mode")
            sys.exit(1)
        run_demo_with_custom_audio(
            audio_path=args.audio,
            text=args.text,
            checkpoint=args.checkpoint,
            phoneme_map_path=args.phoneme_map
        )
    elif args.mode == "participant":
        run_demo_with_participant(
            checkpoint=args.checkpoint,
            phoneme_map_path=args.phoneme_map,
            audio_filename=args.audio
        )


if __name__ == "__main__":
    main()
