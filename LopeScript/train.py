"""
Training script for perceived-only pronunciation assessment model.

This script:
  - loads a minimal Config (perceived-only)
  - builds the model, loss, trainer
  - builds train/validation DataLoaders
  - runs the training loop and saves the best checkpoint
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import get_config
from Model.model import Model
from Model.loss import UnifiedLoss, FocalCTCLoss
from Train.trainer import ModelTrainer
from LopeScript.DataProc.preprocessing import DatasetProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train pronunciation model (perceived-only)")

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training labels/data file (e.g., train_labels.json).",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation labels/data file (e.g., val_labels.json).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (if not set, use config.num_epochs).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (if not set, use config.batch_size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (if not set, use config.device).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="unified",
        choices=["unified", "focal"],
        help="Loss function type: 'unified' (CTC) or 'focal' (FocalCTCLoss)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional).",
    )

    return parser.parse_args()


def build_dataloaders(config, args):
    """
    Build train/validation DataLoaders.

    Dataset은 반드시 다음 key를 가진 batch dict를 반환해야 함:
      - 'waveforms': [B, T_raw]
      - 'audio_lengths': [B]
      - 'perceived_labels': target indices
      - 'perceived_lengths': [B]
    """

    train_dataset = DatasetProcessor(
        labels_path=args.train_data,
        sampling_rate=config.sampling_rate,
        max_length=config.max_length,
        mode="train",
    )

    val_dataset = DatasetProcessor(
        labels_path=args.val_data,
        sampling_rate=config.sampling_rate,
        max_length=config.max_length,
        mode="val",
    )

    collate_fn = getattr(train_dataset, "collate_fn", None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def build_loss(args, config):
    """
    Select loss function based on args.loss_type.
    - unified: UnifiedLoss (단순 CTC 버전)
    - focal:   FocalCTCLoss (CTC + focal weighting)
    """
    if args.loss_type == "focal":
        print("[Loss] Using FocalCTCLoss")
        return FocalCTCLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            blank=config.blank_id,
        )
    else:
        print("[Loss] Using UnifiedLoss (CTC)")
        return UnifiedLoss()


def save_checkpoint(save_dir, epoch, model, best_val_loss):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.pt")

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved best model to {ckpt_path}")


def main():
    args = parse_args()

    # 1) Load base config
    config = get_config()

    # CLI에서 epoch / batch_size / device override 허용
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
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
    model.to(device)

    # 3) Loss + Trainer
    criterion = build_loss(args, config)
    trainer = ModelTrainer(
        model=model,
        config=config,
        device=device,
        logger=None,
    )

    # 4) DataLoaders
    train_loader, val_loader = build_dataloaders(config, args)

    # 5) Resume (optional)
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state"])
        best_val_loss = state.get("best_val_loss", float("inf"))
        start_epoch = state.get("epoch", 0) + 1
        print(f"[Resume] Loaded checkpoint from {args.resume}")
        print(f"[Resume] Start from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # 6) Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n===== Epoch {epoch} / {config.num_epochs} =====")

        train_loss = trainer.train_epoch(
            dataloader=train_loader,
            criterion=criterion,
            epoch=epoch,
        )
        print(f"[Train] Loss: {train_loss:.4f}")

        val_loss = trainer.validate_epoch(
            dataloader=val_loader,
            criterion=criterion,
        )
        print(f"[Valid] Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_dir=args.save_dir,
                epoch=epoch,
                model=model,
                best_val_loss=best_val_loss,
            )

    print("\n[Done] Training finished.")
    print(f"[Done] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
