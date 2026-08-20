"""Ablations that answer 'what did this component actually contribute'.

The naive ablation — retrain with audio removed — answers a subtly different
question than most people intend. Removing an input changes the architecture,
the parameter count, and the optimization problem. A drop in accuracy might be
the missing information, or it might be that the smaller model trains
differently.

Three ablation types here, answering three different questions:

  RETRAIN   — train from scratch without the modality. Answers "how good is a
              model that never had this?" Clean, expensive, and it conflates
              information loss with architecture change.

  ZERO      — keep the trained model, feed zeros for the modality. Answers
              "how much does the trained model rely on this at inference?"
              Cheap, but zeros are out-of-distribution: the model has never
              seen them, so some of the drop is confusion rather than lost
              information.

  SHUFFLE   — feed the modality from a DIFFERENT clip. This is the control
              that makes the others interpretable. The features stay
              in-distribution — real audio, real statistics — but carry no
              information about THIS clip. A drop under shuffle is genuinely
              attributable to the modality's information content.

Shuffle is the one to trust, and it is the one usually missing from ablation
tables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class AblationResult:
    name: str
    accuracy: float
    delta_vs_full: float

    def row(self) -> str:
        return f"{self.name:<22} {self.accuracy:.3f}   {self.delta_vs_full:+.3f}"


def evaluate(model, video: torch.Tensor, audio: torch.Tensor,
             labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        scores = model(video, audio)
        predicted = (torch.sigmoid(scores) >= 0.5).long()
    return float((predicted == labels).float().mean())


def shuffle_modality(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute a modality across the batch, breaking its pairing with labels.

    A derangement rather than a plain permutation: every clip must receive
    features from a DIFFERENT clip. A plain shuffle leaves some clips paired
    with their own features by chance — about 37% of them for large batches
    stay in place under a random permutation in expectation of one fixed point,
    but any fixed points at all weaken the control, since those clips are not
    actually ablated.
    """
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)

    # Fix any position that mapped to itself by rotating it with its neighbour.
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()

    return x[perm]


def run_ablations(model, video: torch.Tensor, audio: torch.Tensor,
                  labels: torch.Tensor, seed: int = 0) -> list[AblationResult]:
    generator = torch.Generator().manual_seed(seed)

    full = evaluate(model, video, audio, labels)
    results = [AblationResult("full model", full, 0.0)]

    zero_audio = evaluate(model, video, torch.zeros_like(audio), labels)
    results.append(AblationResult("audio zeroed", zero_audio, zero_audio - full))

    zero_video = evaluate(model, torch.zeros_like(video), audio, labels)
    results.append(AblationResult("video zeroed", zero_video, zero_video - full))

    # The controls. Averaged over several permutations because a single
    # shuffle is one sample of a random quantity, and reporting one draw as
    # though it were a measurement is how noise gets published as an effect.
    shuffled_audio_scores = [
        evaluate(model, video, shuffle_modality(audio, generator), labels)
        for _ in range(5)
    ]
    shuffled_audio = float(np.mean(shuffled_audio_scores))
    results.append(
        AblationResult("audio shuffled", shuffled_audio, shuffled_audio - full)
    )

    shuffled_video_scores = [
        evaluate(model, shuffle_modality(video, generator), audio, labels)
        for _ in range(5)
    ]
    shuffled_video = float(np.mean(shuffled_video_scores))
    results.append(
        AblationResult("video shuffled", shuffled_video, shuffled_video - full)
    )

    return results


def format_table(results: list[AblationResult]) -> str:
    header = f"{'ablation':<22} {'acc':<7} {'delta'}"
    return "\n".join([header, "-" * 40] + [r.row() for r in results])
