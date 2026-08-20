"""Tests for the failures that do not announce themselves.

Most bugs in an ML pipeline do not raise. They produce a number that is wrong
in a plausible direction, and the pipeline runs to completion. The tests here
target that class specifically — every one of them corresponds to a defect
that would otherwise ship as a believable result.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from data.splits import Clip, build_generator_split
from eval.ablate import shuffle_modality
from eval.threshold import sweep, select_for_max_fpr
from features.extract import ClipFeatures


def make_clips() -> list[Clip]:
    """A fixture where subjects do NOT span every generator.

    This detail is load-bearing. If every subject appeared in every
    generator's output, reserving held-out subjects for test would pull every
    synthetic clip into test and leave training with real clips only — a
    single-class training set, which is unusable.

    That is not merely a fixture concern. It is a real constraint on how such
    a dataset must be assembled: subject coverage has to be partitioned across
    generators, or a subject-disjoint split cannot exist. Here subjects 0-4
    belong to gen_A/gen_B, 5-9 to gen_C, and 10-14 to gen_D.
    """
    clips = []
    subject_ranges = {
        "gen_A": range(0, 5),
        "gen_B": range(0, 5),
        "gen_C": range(5, 10),
        "gen_D": range(10, 15),
    }

    for gen, subjects in subject_ranges.items():
        for i in subjects:
            clips.append(Clip(
                clip_id=f"{gen}_{i}", path=f"/data/{gen}_{i}.mp4", label=1,
                generator=gen, subject_id=f"subject_{i}", duration_s=5.0,
            ))

    for i in range(20):
        clips.append(Clip(
            clip_id=f"real_{i}", path=f"/data/real_{i}.mp4", label=0,
            generator=None, subject_id=f"subject_{i}", duration_s=5.0,
        ))
    return clips


class TestSplitLeakage:
    """Each test here corresponds to a leak that inflates the headline number."""

    def test_held_out_generator_never_in_train(self):
        split = build_generator_split(make_clips(), held_out=("gen_D",))

        train_generators = {c.generator for c in split.train}
        assert "gen_D" not in train_generators

    def test_held_out_generator_appears_in_test(self):
        """Guards the opposite failure: a split so strict that test is empty.

        A split that holds out a generator and also drops it from test would
        pass the leakage test above while measuring nothing at all.
        """
        split = build_generator_split(make_clips(), held_out=("gen_D",))

        test_generators = {c.generator for c in split.test}
        assert "gen_D" in test_generators

    def test_subjects_do_not_span_train_and_test(self):
        """The subtle leak from step 1.

        A model that has seen a subject in training can recognize them at test
        time, inflating accuracy for a reason unrelated to generators.
        """
        split = build_generator_split(make_clips(), held_out=("gen_D",))

        train_subjects = {c.subject_id for c in split.train}
        test_subjects = {c.subject_id for c in split.test}

        assert train_subjects.isdisjoint(test_subjects)

    def test_split_is_deterministic(self):
        """Two calls on the same input must produce identical splits.

        Without this, an accuracy difference between two experiments could be
        the change under test or could be a different split, and there is no
        way to tell which.
        """
        clips = make_clips()
        a = build_generator_split(clips, held_out=("gen_D",))
        b = build_generator_split(clips, held_out=("gen_D",))

        assert [c.clip_id for c in a.train] == [c.clip_id for c in b.train]
        assert [c.clip_id for c in a.test] == [c.clip_id for c in b.test]

    def test_training_retains_both_classes(self):
        """Subject reservation must not empty one class out of training.

        The failure this catches is easy to create and produces no error.
        Subject reservation is transitive through the real clips: reserving a
        held-out generator's subjects also reserves every OTHER clip depicting
        those subjects. If subject coverage overlaps heavily across
        generators, that cascade pulls all synthetic clips into test and
        leaves training with real clips only.

        A single-class training set does not raise. The model trains, the loss
        decreases toward predicting one class always, and accuracy equals the
        majority class rate — a plausible-looking number from a model that
        learned nothing.
        """
        split = build_generator_split(make_clips(), held_out=("gen_D",))

        train_labels = {c.label for c in split.train}
        assert train_labels == {0, 1}, (
            f"training set has only labels {train_labels}; subject reservation "
            "has removed an entire class"
        )


class TestFeatureAlignment:
    def test_mismatched_frame_counts_raise(self):
        """The bug that silently destroys the desync signal."""
        with pytest.raises(ValueError, match="misaligned"):
            ClipFeatures(
                clip_id="c1",
                video=np.zeros((100, 8), dtype=np.float32),
                audio=np.zeros((97, 4), dtype=np.float32),
            )

    def test_empty_features_raise(self):
        with pytest.raises(ValueError, match="no frames"):
            ClipFeatures(
                clip_id="c1",
                video=np.zeros((0, 8), dtype=np.float32),
                audio=np.zeros((0, 4), dtype=np.float32),
            )


class TestAblationControl:
    def test_shuffle_leaves_no_clip_with_its_own_features(self):
        """The derangement property.

        A fixed point means that clip was never ablated, which biases the
        ablation toward understating the modality's contribution.
        """
        x = torch.arange(50, dtype=torch.float32).reshape(50, 1)
        generator = torch.Generator().manual_seed(0)

        shuffled = shuffle_modality(x, generator)

        assert not torch.any(shuffled == x)

    def test_shuffle_preserves_the_multiset(self):
        """Shuffling must permute, not modify.

        If the shuffle altered values, the ablation would no longer be
        in-distribution and would collapse into the zeroing case it exists to
        improve on.
        """
        x = torch.randn(40, 3)
        generator = torch.Generator().manual_seed(0)

        shuffled = shuffle_modality(x, generator)

        assert torch.allclose(shuffled.sum(dim=0), x.sum(dim=0), atol=1e-5)
        assert shuffled.shape == x.shape


class TestThresholdSelection:
    def _scores_and_labels(self):
        rng = np.random.default_rng(0)
        real = rng.normal(0.3, 0.15, 200)
        synthetic = rng.normal(0.7, 0.15, 200)
        scores = np.concatenate([real, synthetic])
        labels = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)
        return scores, labels

    def test_selected_point_respects_the_cap(self):
        scores, labels = self._scores_and_labels()
        point = select_for_max_fpr(sweep(scores, labels), max_fpr=0.05)
        assert point.false_positive_rate <= 0.05

    def test_unachievable_cap_raises(self):
        """Must not silently return a point that violates the requirement.

        A negative cap is used rather than 0.0, because 0.0 is achievable on
        almost any data: the highest threshold flags nothing, so there are no
        false positives and the FPR is exactly zero. A test using 0.0 would
        pass only by accident on data where it happens to be unreachable, and
        would silently stop testing anything the moment the fixture changed.
        """
        scores, labels = self._scores_and_labels()

        with pytest.raises(ValueError, match="no threshold achieves"):
            select_for_max_fpr(sweep(scores, labels), max_fpr=-0.001)

    def test_zero_fpr_is_achievable_and_returns_low_recall(self):
        """The companion to the test above: 0.0 IS reachable, at a cost.

        Pinning this makes the previous test's reasoning explicit rather than
        leaving a reader to wonder why the cap is negative.
        """
        scores, labels = self._scores_and_labels()
        point = select_for_max_fpr(sweep(scores, labels), max_fpr=0.0)

        assert point.false_positive_rate == 0.0
        assert point.recall < 1.0

    def test_higher_cap_gives_at_least_as_much_recall(self):
        """A monotonicity property that must hold for any correct selection.

        Relaxing the false positive budget can never reduce achievable recall.
        This catches selection bugs that a single-threshold test would not,
        because it constrains the relationship between results rather than any
        one value.
        """
        scores, labels = self._scores_and_labels()
        points = sweep(scores, labels)

        strict = select_for_max_fpr(points, max_fpr=0.01)
        relaxed = select_for_max_fpr(points, max_fpr=0.10)

        assert relaxed.recall >= strict.recall
