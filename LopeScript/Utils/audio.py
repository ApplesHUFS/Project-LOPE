"""
Audio utility functions for pronunciation assessment.

This module provides:
    - create_attention_mask: mask valid audio samples for Wav2Vec2
    - compute_output_lengths: raw waveform length → encoder output length
    - enable_specaugment: toggle SpecAugment for Wav2Vec2

These functions are used by both Trainer and Evaluation.
"""

import torch
from torch import Tensor


def create_attention_mask(
    waveforms: Tensor,
    normalized_lengths: Tensor,
) -> Tensor:
    """
    Creates an attention mask indicating valid audio regions.

    Args:
        waveforms: Tensor of shape [B, T_raw]
        normalized_lengths: Tensor [B] where each value = (actual_length / T_raw)

    Returns:
        attention_mask: Tensor [B, T_raw] with 1 for valid frames and 0 for padding.
    """
    batch_size, max_len = waveforms.shape
    device = waveforms.device

    # Convert normalized lengths back to sample counts
    valid_lengths = (normalized_lengths * max_len).round().long()

    attention_mask = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.long,
        device=device,
    )

    for i in range(batch_size):
        L = int(valid_lengths[i].item())
        L = max(1, min(L, max_len))  # Safe clamp
        attention_mask[i, :L] = 1

    return attention_mask


def compute_output_lengths(model, audio_lengths: Tensor) -> Tensor:
    """
    Computes encoder output lengths from raw audio lengths.

    Wav2Vec2Model has an internal function:
        _get_feat_extract_output_lengths

    Args:
        model: Model containing model.encoder.wav2vec2
        audio_lengths: Tensor [B]

    Returns:
        output_lengths: Tensor [B]
    """
    if not (hasattr(model, "encoder") and hasattr(model.encoder, "wav2vec2")):
        raise AttributeError(
            "compute_output_lengths: model.encoder.wav2vec2 not found!"
        )

    wav2vec = model.encoder.wav2vec2

    # Wav2Vec2 API returns Python ints → convert to tensor
    out_lengths = wav2vec._get_feat_extract_output_lengths(
        audio_lengths.cpu()
    )
    return torch.tensor(out_lengths, device=audio_lengths.device, dtype=torch.long)


def enable_specaugment(
    model,
    enable: bool,
    time_prob: float = 0.05,
    feature_prob: float = 0.0,
    time_mask_width: int = 10,
    feature_mask_width: int = 64,
):
    """
    Toggles SpecAugment on the Wav2Vec2 encoder.

    Args:
        model: Model with encoder.wav2vec2
        enable: True → turn on SpecAugment, False → turn it off
    """
    if not (hasattr(model, "encoder") and hasattr(model.encoder, "wav2vec2")):
        return  # If other encoders are used, silently ignore

    wav2vec = model.encoder.wav2vec2
    cfg = wav2vec.config

    if enable:
        cfg.mask_time_prob = time_prob
        cfg.mask_time_length = time_mask_width
        cfg.mask_feature_prob = feature_prob
        cfg.mask_feature_length = feature_mask_width
    else:
        cfg.mask_time_prob = 0.0
        cfg.mask_feature_prob = 0.0
