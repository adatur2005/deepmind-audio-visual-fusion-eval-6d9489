"""Audio and video feature extraction, aligned on a common timeline.

The requirement that shapes everything here: a fusion model needs to compare
what the mouth is doing at time t with what the voice is doing at time t. That
comparison is meaningless unless both streams are on the same clock.

Video arrives at 25 frames per second. Audio arrives at 16,000 samples per
second. Neither divides neatly into the other, so alignment is a decision
rather than something that happens automatically.

The decision here: resample both to a common 25 Hz timeline, with each audio
frame summarizing the 40ms window centred on its video frame. Video is the
coarser stream, so aligning to it avoids inventing video detail that does not
exist.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_FPS = 25
AUDIO_SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // TARGET_FPS  # 640 samples = 40ms


@dataclass(frozen=True, slots=True)
class ClipFeatures:
    """Aligned features. Both arrays have the same first dimension.

    That invariant is checked on construction rather than assumed, because a
    mismatch produces a model that silently compares frame t of video with
    frame t+3 of audio. The loss still decreases, the model still trains, and
    the desync signal in step 3 is destroyed — with nothing indicating a
    problem.
    """

    clip_id: str
    video: np.ndarray  # (T, video_dim)
    audio: np.ndarray  # (T, audio_dim)

    def __post_init__(self) -> None:
        if self.video.shape[0] != self.audio.shape[0]:
            raise ValueError(
                f"{self.clip_id}: misaligned features — "
                f"video has {self.video.shape[0]} frames, "
                f"audio has {self.audio.shape[0]}"
            )
        if self.video.shape[0] == 0:
            raise ValueError(f"{self.clip_id}: no frames extracted")

    @property
    def num_frames(self) -> int:
        return self.video.shape[0]


def extract_audio(path: str) -> np.ndarray:
    """Decode audio to mono 16 kHz float32 via ffmpeg.

    Going through ffmpeg rather than a Python audio library is deliberate: the
    dataset contains several container formats and codecs, and ffmpeg handles
    all of them identically. A library that supports most of them means a
    subset of clips fail in a way that correlates with their source — and
    since source correlates with generator, dropping those clips biases the
    dataset in exactly the dimension being measured.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
            "-loglevel", "error", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def frame_audio(samples: np.ndarray, num_video_frames: int) -> np.ndarray:
    """Summarize audio into one feature vector per video frame.

    Truncates to the video length rather than padding. A clip whose audio runs
    slightly longer than its video is normal — container padding, encoder
    flush — and inventing video frames to match would create silent frames
    with no corresponding image, which the desync detector would then read as
    a genuine mismatch.
    """
    features = np.zeros((num_video_frames, 4), dtype=np.float32)

    for t in range(num_video_frames):
        start = t * SAMPLES_PER_FRAME
        window = samples[start:start + SAMPLES_PER_FRAME]

        if window.size == 0:
            continue

        rms = float(np.sqrt(np.mean(window**2)))
        zero_crossings = float(np.mean(np.abs(np.diff(np.sign(window)))) / 2.0)
        peak = float(np.max(np.abs(window)))
        centroid = float(np.mean(np.abs(np.fft.rfft(window))))

        features[t] = (rms, zero_crossings, peak, centroid)

    return features


def normalize_per_clip(features: np.ndarray) -> np.ndarray:
    """Standardize each feature dimension within the clip.

    Per-clip rather than per-dataset, and the reason is a leakage one again.

    Dataset-level statistics computed over all clips would include the test
    clips, letting information about the test distribution into training. It
    is a small leak, and it is the kind that survives review because computing
    global statistics feels like preprocessing rather than like training.

    Per-clip normalization also removes recording-level differences —
    microphone gain, overall brightness — that correlate with source and would
    otherwise be another shortcut for the model to take.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def cache_key(path: str) -> str:
    """Content hash for the feature cache.

    Keyed on file CONTENT, not path or mtime. A re-encoded clip at the same
    path with the same timestamp would otherwise return stale features, and
    stale features are undetectable downstream — the arrays have the right
    shape and plausible values.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
