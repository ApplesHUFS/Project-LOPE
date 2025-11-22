"""Unified model architecture for pronunciation assessment.

This module implements the main model combining Wav2Vec2 encoder,
feature processing, and task-specific output heads with automatic
architecture adaptation and model factory pattern.
"""

import torch.nn as nn
from transformers import Wav2Vec2Config

from .encoder import Wav2VecEncoder, BaseEncoder
from .head import PhonemeHead

class Model(nn.Module):
  """Unified model for multitask pronunciation assessment.
  
  Architecture:
    1. Wav2Vec2 encoder: Extracts audio features
    2. Feature encoder: base
  
  The model automatically adapts its dimensions based on the pretrained
  Wav2Vec2 model configuration.
  """
  
  def __init__(
      self,
      pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53",
      num_phonemes: int = 42,
      dropout: float = 0.1
  ):
    """Initializes unified model.

    Args:
      pretrained_model_name: Pretrained Wav2Vec2 model name.
      num_phonemes: Number of phoneme classes.
      dropout: Dropout probability.
    """
    super().__init__()

    # Wav2Vec2 audio encoder
    self.encoder = Wav2VecEncoder(pretrained_model_name)

    # Get Wav2Vec2 output dimension from configuration
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    wav2vec_dim = config.hidden_size

    # Create feature encoder using factory pattern
    self.feature_encoder = BaseEncoder(wav2vec_dim, wav2vec_dim, dropout),

    # Determine output dimension based on model type
    output_dim = wav2vec_dim

    self.pred_head = PhonemeHead(
        output_dim,
        num_phonemes,
        dropout
    )

  def forward(
      self,
      waveform,
      attention_mask=None,
  ):
    """Forward pass through the model.

    Args:
      waveform: Input audio of shape [batch_size, seq_len].
      attention_mask: Attention mask of shape [batch_size, seq_len].

    Returns:
      Dictionary containing logits from active heads.
    """
    # Extract audio features
    features = self.encoder(waveform, attention_mask)

    # Process features through feature encoder
    processed_features = self.feature_encoder(features)

    # perceived phoneme prediction
    outputs = self.pred_head(processed_features)

    return outputs