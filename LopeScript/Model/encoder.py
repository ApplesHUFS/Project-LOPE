"""
Encoder architectures for pronunciation assessment.

This module implements audio encoders including:

1. Wav2Vec2-based encoder (Wav2VecEncoder)
2. Simple feed-forward encoder (FeedForwardEncoder)
3. Transformer-based encoder (TransformerEncoder)
4. Minimal baseline encoder (BaseEncoder)

All encoders take features of shape [batch_size, seq_len, feature_dim]
and return the same shape, unless otherwise noted.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


# ---------------------------------------------------------------------------
# 1. Wav2Vec2-based audio encoder
# ---------------------------------------------------------------------------

class Wav2VecEncoder(nn.Module):
    """
    Wav2Vec2-based audio encoder for robust feature extraction.

    기본 설정은 인터넷 없이도 돌아가도록 **랜덤 초기화 Wav2Vec2** 를 사용한다.
    추후 HuggingFace pretrained 모델을 쓰고 싶으면 use_pretrained=True 로 바꾸면 됨.

    Args:
        pretrained_model_name:
            (optional) HuggingFace model name or local path.
            예: "facebook/wav2vec2-large-xlsr-53"
        use_pretrained:
            True이면 HuggingFace에서 from_pretrained()로 로드 (인터넷 필요).
            False이면 Wav2Vec2Config()로 랜덤 초기화 (기본값, 안전 모드).
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53",
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.pretrained_model_name = pretrained_model_name
        self.use_pretrained = use_pretrained

        if self.use_pretrained:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        else:
            config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
            self.wav2vec2 = Wav2Vec2Model(config)

        self.wav2vec2.config.mask_time_prob = 0.0
        self.wav2vec2.config.mask_feature_prob = 0.0

        self.output_dim = self.wav2vec2.config.hidden_size

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes raw waveform to contextualized features.

        Args:
            waveform:
                Float tensor of shape [batch_size, seq_len], 16kHz audio.
            attention_mask:
                Optional mask of shape [batch_size, seq_len].

        Returns:
            features:
                Tensor of shape [batch_size, seq_len_subsampled, hidden_dim].
        """
        outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
        # last_hidden_state: [B, T', H]
        return outputs.last_hidden_state


# ---------------------------------------------------------------------------
# 2. Simple feed-forward encoder
# ---------------------------------------------------------------------------

class FeedForwardEncoder(nn.Module):
    """
    Simple feed-forward encoder on top of frame-level features.

    입력:  [B, T, D_in]
    출력:  [B, T, D_out]

    여러 층의 Linear + ReLU + Dropout 으로 구성된 간단한 인코더.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        dim_in = input_dim
        for i in range(num_layers):
            dim_out = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            dim_in = dim_out

        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, D_in]

        Returns:
            [B, T, D_out]
        """
        # nn.Sequential 은 마지막 차원(D) 에 Linear 를 적용하므로 그대로 사용 가능
        return self.net(features)


# ---------------------------------------------------------------------------
# 3. Transformer-based encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for contextual feature enhancement.

    입력:  [B, T, D]
    출력:  [B, T, D]

    Args:
        input_dim:   입력 feature 차원
        num_layers:  Transformer encoder layer 수
        num_heads:   multi-head attention head 수
        dim_feedforward: FFN 내부 차원
        dropout:     dropout 비율
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, T, D] 형태 사용
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.output_dim = input_dim

    def forward(
        self,
        features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features:
                [B, T, D] tensor.
            src_key_padding_mask:
                Optional bool mask [B, T], True = padding 위치.

        Returns:
            [B, T, D] tensor.
        """
        x = self.layer_norm(features)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# 4. Minimal baseline encoder (기존 BaseEncoder 대체)
# ---------------------------------------------------------------------------

class BaseEncoder(nn.Module):
    """
    Base encoder with minimal processing for baseline models.

    - LayerNorm + Dropout 만 적용
    - 차원 변화 없음 (input_dim == hidden_dim)

    보통은 Wav2VecEncoder 의 출력에 바로 CTC head 를 얹고 싶을 때,
    혹은 복잡한 encoder 대신 아주 간단한 baseline 을 쓰고 싶을 때 사용.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert (
            input_dim == hidden_dim
        ), "BaseEncoder requires input_dim == hidden_dim"

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, D]

        Returns:
            [B, T, D]
        """
        return self.dropout(self.layer_norm(features))
