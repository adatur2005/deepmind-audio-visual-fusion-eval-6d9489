"""Building splits that measure generalization rather than memorization.

The central decision: split by GENERATOR, not by clip.

A random clip-level split puts clips from the same generator on both sides. The
model then sees that generator during training and is tested on more of its
output, which measures whether it memorized one generator's artifacts. That
number is high and it does not transfer.

A generator-level split holds out entire generators. Training never sees them,
so test accuracy answers the question that actually matters: does this work on
something new?

The cost is that the number will be much lower and much noisier. That is not a
defect of the method — it is the honest difficulty of the problem becoming
visible.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    path: str
    label: int            # 1 = synthetic, 0 = real
    generator: str | None  # None for real clips
    subject_id: str        # the person depicted
    duration_s: float


@dataclass(frozen=True, slots=True)
class Split:
    train: tuple[Clip, ...]
    val: tuple[Clip, ...]
    test: tuple[Clip, ...]
    held_out_generators: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"train {len(self.train)} / val {len(self.val)} / test {len(self.test)}\n"
            f"held-out generators: {', '.join(self.held_out_generators)}"
        )


def build_generator_split(
    clips: list[Clip],
    held_out: tuple[str, ...],
) -> Split:
    """Split so that held-out generators appear ONLY in test.

    Two leakage paths are closed here, and the second is the subtle one.

    1. Generator leakage — the point of the whole function. No clip from a
       held-out generator may appear in training.

    2. Subject leakage — if the same PERSON appears in both training and test,
       the model can learn that individual's face and voice rather than
       learning synthesis artifacts. It then scores well on test subjects it
       has memorized, and the number is inflated for a reason that has nothing
       to do with generators at all.

       This one is easy to miss because the generator split looks correct
       while it is happening.
    """
    held_out_set = set(held_out)

    # Subjects appearing in any held-out-generator clip are reserved for test.
    test_subjects: set[str] = {
        c.subject_id for c in clips
        if c.generator is not None and c.generator in held_out_set
    }

    train_pool: list[Clip] = []
    test_pool: list[Clip] = []

    for c in clips:
        is_held_out_generator = c.generator is not None and c.generator in held_out_set
        is_reserved_subject = c.subject_id in test_subjects

        if is_held_out_generator or is_reserved_subject:
            test_pool.append(c)
        else:
            train_pool.append(c)

    # Validation comes out of the TRAINING generators, not the held-out ones.
    #
    # This is deliberate and it is a real cost. Validation accuracy will be
    # optimistic — it measures in-distribution performance — so it cannot be
    # used to estimate final performance.
    #
    # But using held-out generators for validation would mean selecting
    # hyperparameters against the test distribution, which is the same leakage
    # in a slower form: after enough tuning runs, the test set has effectively
    # been trained on. Optimistic validation is the lesser problem, provided
    # nobody quotes it as the result.
    train_pool.sort(key=lambda c: c.clip_id)
    cut = int(len(train_pool) * 0.85)

    return Split(
        train=tuple(train_pool[:cut]),
        val=tuple(train_pool[cut:]),
        test=tuple(test_pool),
        held_out_generators=held_out,
    )


def load_manifest(path: str) -> list[Clip]:
    clips: list[Clip] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            clips.append(Clip(**raw))
    return clips


def generator_counts(clips: list[Clip]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for c in clips:
        counts[c.generator or "real"] += 1
    return dict(counts)
