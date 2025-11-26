"""Evaluation script for pronunciation assessment model."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .config import get_config
from .Model.model import Model
from .Model.loss import FocalCTCLoss
from .Evaluation.eval import evaluate
from dataset import PronunciationDataset, collate_batch


def parse_args():
  parser = argparse.ArgumentParser(description="Evaluate pronunciation model")

  parser.add_argument(
      "--test_data",
      type=str,
      required=True,
      help="Path to test data JSON file",
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      required=True,
      help="Path to trained model checkpoint",
  )
  parser.add_argument(
      "--phoneme_map",
      type=str,
      default="data/phoneme_to_id.json",
      help="Path to phoneme to ID mapping JSON",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=None,
      help="Evaluation batch size",
  )
  parser.add_argument(
      "--device",
      type=str,
      default=None,
      help="Device to use: cuda or cpu",
  )
  parser.add_argument(
      "--with_loss",
      action="store_true",
      help="Also compute CTC loss on test set",
  )

  return parser.parse_args()


def build_dataloader(config, args, phoneme_to_id):
  """Build test DataLoader."""

  test_dataset = PronunciationDataset(
      json_path=args.test_data,
      phoneme_to_id=phoneme_to_id,
      training_mode=config.training_mode,
      max_length=config.max_length,
      sampling_rate=config.sampling_rate,
      device=config.device,
  )

  def collate_fn(batch):
    return collate_batch(batch, config.training_mode)

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

  config = get_config()

  if args.batch_size is not None:
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

  if not os.path.isfile(args.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

  state = torch.load(args.checkpoint, map_location="cpu")
  if "model_state" in state:
    model.load_state_dict(state["model_state"])
  else:
    model.load_state_dict(state)

  test_loader = build_dataloader(config, args, phoneme_to_id)

  criterion = None
  if args.with_loss:
    criterion = FocalCTCLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        blank=config.blank_id,
    )

  metrics = evaluate(
      model=model,
      dataloader=test_loader,
      device=device,
      blank_id=config.blank_id,
      criterion=criterion,
      use_specaugment=False,
  )

  print("\n=== Evaluation Results ===")
  for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


if __name__ == "__main__":
  main()
