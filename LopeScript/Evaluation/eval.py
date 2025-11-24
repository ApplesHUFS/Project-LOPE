"""
Evaluation utilities for pronunciation assessment model.

This module provides functions to:
  - run the model on a dataloader
  - perform CTC greedy decoding
  - compute corpus-level PER
  - (optionally) compute CTC loss during evaluation
"""

from typing import Dict, List, Sequence, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.audio import (
    create_attention_mask,
    compute_output_lengths,
    enable_specaugment,
)
from .metric import corpus_per


def greedy_ctc_decode(
    logits: Tensor,
    blank_id: int = 0
) -> List[List[int]]:
    """Greedy CTC decoding.

    Args:
        logits: Model output logits of shape [batch, time, num_classes].
        blank_id: Index of the CTC blank token.

    Returns:
        List of length batch_size; each element is a list of predicted
        token ids for that utterance (after collapsing repeats and
        removing blanks).
    """
    # [B, T, C] -> [B, T]
    pred_ids = torch.argmax(logits, dim=-1)  # argmax over classes

    batch_size, max_time = pred_ids.shape
    sequences: List[List[int]] = []

    for b in range(batch_size):
        seq: List[int] = []
        prev_id: Optional[int] = None

        for t in range(max_time):
            token_id = int(pred_ids[b, t].item())

            # Skip blanks
            if token_id == blank_id:
                prev_id = None
                continue

            # Collapse consecutive duplicates
            if prev_id is not None and token_id == prev_id:
                continue

            seq.append(token_id)
            prev_id = token_id

        sequences.append(seq)

    return sequences


def _prepare_reference_sequences(
    targets: Tensor,
    lengths: Tensor
) -> List[List[int]]:
    """Converts CTC targets and lengths into a list of sequences.

    Supports both:
      - padded 2D format [batch, max_target_len]
      - concatenated 1D format [sum(target_lengths)]

    Args:
        targets: Target tensor (1D or 2D).
        lengths: Lengths tensor of shape [batch].

    Returns:
        List of length batch_size; each element is a list of target ids.
    """
    lengths_list = lengths.cpu().tolist()

    if targets.dim() == 2:
        # [B, S] padded format
        return [
            targets[i, :lengths_list[i]].cpu().tolist()
            for i in range(targets.size(0))
        ]

    elif targets.dim() == 1:
        # [sum(S_i)] concatenated format
        sequences: List[List[int]] = []
        offset = 0
        for L in lengths_list:
            L_int = int(L)
            seq = targets[offset:offset + L_int].cpu().tolist()
            sequences.append(seq)
            offset += L_int
        return sequences

    else:
        raise ValueError(
            f"Unsupported target tensor shape: {targets.shape}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    *,
    blank_id: int = 0,
    criterion: Optional[nn.Module] = None,
    use_specaugment: bool = False,
) -> Dict[str, float]:
    """Runs evaluation on a dataloader and computes PER (and optionally loss).

    Args:
        model: Trained pronunciation assessment model.
        dataloader: DataLoader providing batches with at least:
            - 'waveforms': [B, T_raw]
            - 'audio_lengths': [B]
            - 'perceived_labels': target indices (1D or 2D tensor)
            - 'perceived_lengths': [B]
        device: Device to run evaluation on ('cuda' or 'cpu').
        blank_id: CTC blank index (default 0).
        criterion: Optional CTC-style loss function. If provided, it should
            take arguments:
                criterion(log_probs[T, B, C],
                          targets,
                          input_lengths[B],
                          target_lengths[B])
        use_specaugment: Whether to keep SpecAugment enabled during eval.
            Typically False for dev/test.

    Returns:
        Dictionary with evaluation metrics. Keys:
            - 'per': corpus-level phoneme error rate
            - 'loss': average loss over batches (if criterion is given)
    """
    model.to(device)
    model.eval()
    enable_specaugment(model, use_specaugment)

    all_refs: List[Sequence[int]] = []
    all_hyps: List[Sequence[int]] = []

    total_loss = 0.0
    num_loss_batches = 0

    progress = tqdm(dataloader, desc="Evaluation")

    for batch in progress:
        if batch is None:
            continue

        # 1) Prepare inputs
        waveforms: Tensor = batch["waveforms"].to(device)          # [B, T_raw]
        audio_lengths: Tensor = batch["audio_lengths"].to(device)  # [B]

        # Compute encoder output lengths for CTC
        input_lengths: Tensor = compute_output_lengths(
            model, audio_lengths
        )  # [B]

        normalized_lengths = audio_lengths.float() / waveforms.shape[1]
        attention_mask: Tensor = create_attention_mask(
            waveforms, normalized_lengths
        )  # [B, T_raw]

        # 2) Forward pass
        logits: Tensor = model(waveforms, attention_mask)  # [B, T_enc, C]

        # 3) Optional loss computation (CTC-style)
        if criterion is not None:
            log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, C]
            log_probs = log_probs.transpose(0, 1)          # [T, B, C]

            targets = batch["perceived_labels"].to(device)
            target_lengths = batch["perceived_lengths"].to(device)

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            # FocalCTCLoss / CTCLoss 둘 다 scalar 반환 가정
            total_loss += float(loss.item())
            num_loss_batches += 1

        # 4) Decode predictions (CTC greedy)
        batch_hyps = greedy_ctc_decode(logits, blank_id=blank_id)

        # 5) Prepare reference sequences
        targets_cpu = batch["perceived_labels"]
        target_lengths_cpu = batch["perceived_lengths"]
        batch_refs = _prepare_reference_sequences(
            targets_cpu,
            target_lengths_cpu,
        )

        # 6) Accumulate for corpus-level PER
        all_refs.extend(batch_refs)
        all_hyps.extend(batch_hyps)

    # 7) Compute metrics
    per_value = corpus_per(all_refs, all_hyps)

    metrics: Dict[str, float] = {"per": float(per_value)}

    if criterion is not None and num_loss_batches > 0:
        metrics["loss"] = total_loss / float(num_loss_batches)

    return metrics
