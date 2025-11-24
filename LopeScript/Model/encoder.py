"""Encoder architectures for pronunciation assessment.

This module implements audio encoders including Wav2Vec2-based encoder,
simple feed-forward encoder, and Transformer encoder for feature enhancement.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2VecEncoder(nn.Module):
  """Wav2Vec2-based audio encoder for robust feature extraction.
  
  Uses pretrained Wav2Vec2 model to extract contextualized audio features.
  """
  
  def __init__(self, pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53"):
    """Initializes Wav2Vec2 encoder.
    
    Args:
      pretrained_model_name: Name or path of pretrained Wav2Vec2 model.
    """
    super().__init__()
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    config.mask_time_prob = 0.0
    config.mask_feature_prob = 0.0
    self.wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_model_name,
        config=config
    )

  def forward(self, waveform, attention_mask=None):
    """Encodes audio waveform to features.
    
    Args:
      waveform: Input audio tensor of shape [batch_size, seq_len].
      attention_mask: Optional attention mask of shape [batch_size, seq_len].
      
    Returns:
      Encoded features of shape [batch_size, seq_len, hidden_dim].
    """
    outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
    return outputs.last_hidden_state

class BaseEncoder(nn.Module):
  """Base encoder with minimal processing for baseline models.

  Only applies layer normalization and dropout without additional transformation,
  similar to directly using Wav2Vec2 features for CTC output.
  """

  def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
    """Initializes base encoder.

    Args:
      input_dim: Input feature dimension.
      hidden_dim: Output dimension (must match input_dim for base model).
      dropout: Dropout probability.
    """
    super().__init__()
    assert input_dim == hidden_dim, "Base encoder requires input_dim == hidden_dim"
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, features):
    """Applies minimal processing to features.

    Args:
      features: Input features of shape [batch_size, seq_len, input_dim].

    Returns:
      Processed features of shape [batch_size, seq_len, hidden_dim].
    """
    return self.dropout(self.layer_norm(features))
