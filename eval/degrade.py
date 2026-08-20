"""Degradations that real clips undergo before anyone sees them.

A clip in the wild has been re-encoded by an upload pipeline, cropped for a
different aspect ratio, and possibly screen-recorded. Accuracy on pristine
files is not a prediction of deployed accuracy.

Two of these deserve particular attention:

COMPRESSION is not merely noise. It removes exactly the high-frequency detail
that many synthesis artifacts live in — the subtle boundary inconsistencies
around hair and teeth, the upsampling texture. A detector relying on those
does not degrade gracefully under compression; it stops working. Heavy
compression is therefore both a realistic condition and an effective attack,
which is why it is the first thing to measure.

TEMPORAL CROPPING is subtler and more interesting. Cutting frames from a clip
shifts audio relative to video unless both are cut identically. That
introduces exactly the desync our best model is detecting, so a cropped REAL
clip can look synthetic to it. The degradation mimics the signal, which makes
this the most informative test in the suite.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Degradation:
    name: str
    description: str
    ffmpeg_args: tuple[str, ...]


DEGRADATIONS: tuple[Degradation, ...] = (
    Degradation(
        name="clean",
        description="No degradation. The baseline everything else is read against.",
        ffmpeg_args=(),
    ),
    Degradation(
        name="compress_medium",
        description="H.264 CRF 32 — typical of a social platform re-encode.",
        ffmpeg_args=("-c:v", "libx264", "-crf", "32"),
    ),
    Degradation(
        name="compress_heavy",
        description="H.264 CRF 40 — aggressive, but common after several reuploads.",
        ffmpeg_args=("-c:v", "libx264", "-crf", "40"),
    ),
    Degradation(
        name="downscale",
        description="Halve resolution — mobile upload or a small embedded player.",
        ffmpeg_args=("-vf", "scale=iw/2:ih/2"),
    ),
    Degradation(
        name="crop_face",
        description="Centre crop to 70% — reframing for a vertical feed.",
        ffmpeg_args=("-vf", "crop=iw*0.7:ih*0.7"),
    ),
    Degradation(
        name="temporal_crop",
        description="Drop the first 5 frames of video only — induces desync.",
        ffmpeg_args=("-vf", "select=gte(n\\,5)", "-af", "anull"),
    ),
)


def apply(clip_path: str, degradation: Degradation) -> str:
    """Apply one degradation, returning a path to the transformed clip."""
    if not degradation.ffmpeg_args:
        return clip_path

    out = Path(tempfile.mkdtemp()) / f"{degradation.name}.mp4"
    subprocess.run(
        ["ffmpeg", "-i", clip_path, *degradation.ffmpeg_args,
         "-loglevel", "error", str(out)],
        check=True,
    )
    return str(out)
