"""
Evaluation script for perceived-only pronunciation assessment model.

This script:
  - loads minimal Config
  - builds the model and loads a trained checkpoint
  - builds a test DataLoader
  - runs Evaluation/eval.evaluate to compute PER (+ optional CTC loss)
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import get_config
from Model.model import Model
from Model.loss import FocalCTCLoss  
from Evaluation.eval import evaluate
from LopeScript.DataProc.preprocessing import DatasetProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pronunciation model (perceived-only)")

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test labels/data file (e.g., test_labels.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Evaluation batch size (if not set, use config.eval_batch_size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (if not set, use config.device).",
    )
    parser.add_argument(
        "--with_loss",
        action="store_true",
        help="Also compute CTC/Focal loss on test set.",
    )

    return parser.parse_args()


def build_dataloader(config, args):
    """
    Build test DataLoader.

    Dataset은 반드시 다음 key를 가진 batch dict를 반환해야 함:
      - 'waveforms': [B, T_raw]
      - 'audio_lengths': [B]
      - 'perceived_labels': target indices
      - 'perceived_lengths': [B]
    """
    test_dataset = DatasetProcessor(
        labels_path=args.test_data,
        sampling_rate=config.sampling_rate,
        max_length=config.max_length,
        mode="test",
    )

    collate_fn = getattr(test_dataset, "collate_fn", None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return test_loader


def main():
    args = parse_args()

    # 1) Load config
    config = get_config()

    if args.batch_size is not None:
        config.eval_batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        device = "cpu"

    # 2) Build model
    model = Model(
        pretrained_model_name=config.pretrained_model_name,
        num_phonemes=config.num_phonemes,
        dropout=config.dropout,
    )

    # 3) Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    # 4) Build test DataLoader
    test_loader = build_dataloader(config, args)

    # 5) Loss function (optional)
    criterion = None
    if args.with_loss:
        print("[Eval] Using FocalCTCLoss for test loss.")
        criterion = FocalCTCLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            blank=config.blank_id,
        )

    # 6) Run evaluation (PER + optional loss)
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        blank_id=config.blank_id,
        criterion=criterion,
        use_specaugment=False,  # 보통 테스트에서는 augmentation 끔
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
