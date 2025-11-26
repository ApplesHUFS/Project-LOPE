"""Training script for pronunciation assessment model."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from LopeScript.config import get_config
from .Model.model import Model
from .Model.loss import UnifiedLoss, FocalCTCLoss
from .Train.trainer import ModelTrainer
from dataset import PronunciationDataset, collate_batch


def parse_args():
  parser = argparse.ArgumentParser(description="Train pronunciation model")

  parser.add_argument(
      "--train_data",
      type=str,
      required=True,
      help="Path to training data JSON file",
  )
  parser.add_argument(
      "--val_data",
      type=str,
      required=True,
      help="Path to validation data JSON file",
  )
  parser.add_argument(
      "--phoneme_map",
      type=str,
      default="data/phoneme_to_id.json",
      help="Path to phoneme to ID mapping JSON",
  )
  parser.add_argument(
      "--save_dir",
      type=str,
      default="experiments",
      help="Directory to save checkpoints",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=None,
      help="Number of training epochs",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=None,
      help="Training batch size",
  )
  parser.add_argument(
      "--device",
      type=str,
      default=None,
      help="Device to use: cuda or cpu",
  )
  parser.add_argument(
      "--loss_type",
      type=str,
      default="unified",
      choices=["unified", "focal"],
      help="Loss function type",
  )
  parser.add_argument(
      "--resume",
      type=str,
      default=None,
      help="Path to checkpoint to resume from",
  )

  return parser.parse_args()


def build_dataloaders(config, args, phoneme_to_id):
  """Build train/validation DataLoaders."""

  train_dataset = PronunciationDataset(
      json_path=args.train_data,
      phoneme_to_id=phoneme_to_id,
      training_mode=config.training_mode,
      max_length=config.max_length,
      sampling_rate=config.sampling_rate,
      device=config.device,
  )

  val_dataset = PronunciationDataset(
      json_path=args.val_data,
      phoneme_to_id=phoneme_to_id,
      training_mode=config.training_mode,
      max_length=config.max_length,
      sampling_rate=config.sampling_rate,
      device=config.device,
  )

  def collate_fn(batch):
    return collate_batch(batch, config.training_mode)

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
  """Select loss function based on args.loss_type."""
  if args.loss_type == "focal":
    return FocalCTCLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        blank=config.blank_id,
    )
  else:
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


def main():
  args = parse_args()

  config = get_config()

  if args.epochs is not None:
    config.num_epochs = args.epochs
  if args.batch_size is not None:
    config.batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
  if args.device is not None:
    config.device = args.device

  device = config.device
  if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
    config.device = "cpu"

  with open(args.phoneme_map, 'r') as f:
    phoneme_to_id = json.load(f)

  model = Model(
      pretrained_model_name=config.pretrained_model_name,
      num_phonemes=config.num_phonemes,
      dropout=config.dropout,
  )
  model.to(device)

  criterion = build_loss(args, config)
  trainer = ModelTrainer(
      model=model,
      config=config,
      device=device,
      logger=None,
  )

  train_loader, val_loader = build_dataloaders(config, args, phoneme_to_id)

  start_epoch = 1
  best_val_loss = float("inf")

  if args.resume is not None:
    if not os.path.isfile(args.resume):
      raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    state = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(state["model_state"])
    best_val_loss = state.get("best_val_loss", float("inf"))
    start_epoch = state.get("epoch", 0) + 1

  for epoch in range(start_epoch, config.num_epochs + 1):
    train_loss = trainer.train_epoch(
        dataloader=train_loader,
        criterion=criterion,
        epoch=epoch,
    )

    val_loss = trainer.validate_epoch(
        dataloader=val_loader,
        criterion=criterion,
    )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(
          save_dir=args.save_dir,
          epoch=epoch,
          model=model,
          best_val_loss=best_val_loss,
      )


if __name__ == "__main__":
  main()
