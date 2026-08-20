"""Three ways to combine audio and video, with different expressive power.

The distinction that matters: WHEN the modalities meet determines what
relationships the model can represent.

  EARLY  — concatenate features, then process jointly. Can learn cross-modal
           interactions from the first layer, but a strong single modality can
           dominate and the other is effectively ignored.

  LATE   — process each modality separately, combine the two SCORES. Robust,
           interpretable, and structurally incapable of learning any
           relationship BETWEEN the modalities. See the discussion below.

  CROSS  — each modality attends to the other over time, then combine. Can
           represent "does the mouth match the voice at this moment", which is
           the signal that actually distinguishes a good impersonation.

Late fusion is the important negative case. A synthetic clip can have a
perfectly natural face and a perfectly natural voice that simply do not agree
with each other. Late fusion sees a real-looking face and a real-looking voice,
scores both as real, and combines two confident correct judgements into a
confidently wrong answer. The failure is structural, not a matter of capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """Concatenate along the feature axis, then a temporal encoder."""

    def __init__(self, video_dim: int, audio_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=video_dim + audio_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([video, audio], dim=-1)
        encoded, _ = self.encoder(joint)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class LateFusion(nn.Module):
    """Separate encoders, combine the two scalar scores.

    Kept as a baseline precisely because it CANNOT represent cross-modal
    agreement. Its gap against cross-attention on the desync-heavy generators
    is the experimental evidence that agreement is what carries the signal —
    without this baseline that claim would be an assertion.
    """

    def __init__(self, video_dim: int, audio_dim: int, hidden: int = 128):
        super().__init__()
        self.video_encoder = nn.GRU(video_dim, hidden, batch_first=True,
                                    bidirectional=True)
        self.audio_encoder = nn.GRU(audio_dim, hidden, batch_first=True,
                                    bidirectional=True)
        self.video_head = nn.Linear(hidden * 2, 1)
        self.audio_head = nn.Linear(hidden * 2, 1)
        self.combine = nn.Linear(2, 1)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v, _ = self.video_encoder(video)
        a, _ = self.audio_encoder(audio)

        v_score = self.video_head(v.mean(dim=1))
        a_score = self.audio_head(a.mean(dim=1))

        # The two modalities meet HERE, as two numbers. Everything about their
        # temporal relationship has already been discarded by the pooling
        # above. No amount of capacity in `combine` can recover it.
        return self.combine(torch.cat([v_score, a_score], dim=-1)).squeeze(-1)


class CrossAttentionFusion(nn.Module):
    """Each modality attends to the other across time.

    This is the architecture that can represent synchronization. Video queries
    audio: for each video frame, which audio frames are relevant? In a real
    clip the answer concentrates near the same timestep — lips and voice
    coincide. In a clip where audio was generated separately from video, the
    attention has no consistent temporal structure to find.
    """

    def __init__(self, video_dim: int, audio_dim: int, hidden: int = 128,
                 heads: int = 4):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)

        self.v2a = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.a2v = nn.MultiheadAttention(hidden, heads, batch_first=True)

        self.norm_v = nn.LayerNorm(hidden)
        self.norm_a = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(video)
        a = self.audio_proj(audio)

        # Video attends to audio and vice versa. Both directions, because the
        # asymmetry matters: a generated voice over a real face and a real
        # voice over a generated face are different failure modes, and a
        # single direction would be better at detecting one than the other.
        v_attended, _ = self.v2a(query=v, key=a, value=a)
        a_attended, _ = self.a2v(query=a, key=v, value=v)

        v_out = self.norm_v(v + v_attended)
        a_out = self.norm_a(a + a_attended)

        pooled = torch.cat([v_out.mean(dim=1), a_out.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)


def build(name: str, video_dim: int, audio_dim: int) -> nn.Module:
    if name == "early":
        return EarlyFusion(video_dim, audio_dim)
    if name == "late":
        return LateFusion(video_dim, audio_dim)
    if name == "cross_attention":
        return CrossAttentionFusion(video_dim, audio_dim)
    raise ValueError(f"unknown fusion strategy: {name}")
