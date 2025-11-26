"""Loss functions for pronunciation assessment model.

This module implements Focal CTC Loss for handling class imbalance
and a unified loss for multitask learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class FocalCTCLoss(nn.Module):
  """Focal Loss applied to CTC Loss for class imbalance handling.
  
  Applies focal weighting to CTC loss to focus training on hard examples
  and handle class imbalance.
  """
  
  def __init__(
      self,
      alpha: float = 1.0,
      gamma: float = 2.0,
      blank: int = 0,
      zero_infinity: bool = True
  ):
    """Initializes Focal CTC Loss.
    
    Args:
      alpha: Weight factor for class balancing.
      gamma: Focusing parameter (higher = more focus on hard examples).
      blank: Index of CTC blank token.
      zero_infinity: Whether to set infinite losses to zero.
    """
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.ctc_loss = nn.CTCLoss(
        blank=blank, 
        reduction='none',
        zero_infinity=zero_infinity
    )

  def forward(
      self,
      log_probs: torch.Tensor,
      targets: torch.Tensor,
      input_lengths: torch.Tensor,
      target_lengths: torch.Tensor
  ) -> torch.Tensor:
    """Computes Focal CTC Loss.
    
    Args:
      log_probs: Log probabilities of shape [T, N, C].
      targets: Target sequences of shape [N, S].
      input_lengths: Input sequence lengths of shape [N].
      target_lengths: Target sequence lengths of shape [N].
      
    Returns:
      Scalar loss tensor.
    """
    # Compute CTC loss per sample
    ctc_losses = self.ctc_loss(
        log_probs, 
        targets, 
        input_lengths, 
        target_lengths
    )
    ctc_losses = torch.clamp(ctc_losses, min=1e-6)
    
    # Apply focal weighting
    probability = torch.exp(-ctc_losses)
    probability = torch.clamp(probability, min=1e-6, max=1.0)
    focal_weights = self.alpha * (1 - probability) ** self.gamma
    focal_losses = ctc_losses * focal_weights

    return focal_losses.mean()


class UnifiedLoss(nn.Module):
  """loss function for multitask pronunciation assessment.
  """
  
  def __init__(self):
    """Initializes loss.
    """
    super().__init__()

    self.ctc_loss = nn.CTCLoss(
        blank= 0, 
        reduction='none',
        zero_infinity=True
    )

  def forward(
      self,
      outputs: Dict[str, torch.Tensor],
      perceived_targets: torch.Tensor,
      perceived_input_lengths: torch.Tensor,
      perceived_target_lengths: torch.Tensor,
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes CTC loss for phoneme recognition.

    Args:
      outputs: Model outputs containing logits for each task.
      perceived_targets: Perceived phoneme targets.
      perceived_input_lengths: Perceived input sequence lengths.
      perceived_target_lengths: Perceived target sequence lengths.

    Returns:
      Tuple of (total_loss, loss_components_dict).
    """
    log_probs = torch.log_softmax(outputs['perceived_logits'], dim=-1)
    ctc_losses = self.ctc_loss(
        log_probs.transpose(0, 1),
        perceived_targets,
        perceived_input_lengths,
        perceived_target_lengths
    ).mean()

    return ctc_losses, {'ctc_loss': ctc_losses.item()}