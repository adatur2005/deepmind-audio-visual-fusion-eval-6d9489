# Results

All numbers are on **held-out generators only** — generators absent from
training. In-distribution numbers appear nowhere in this document, because
reporting both invites the higher one to be quoted.

## Fusion strategy, clean clips, unseen generators

| strategy        | accuracy | AUC   | recall @ FPR 0.01 |
|-----------------|----------|-------|-------------------|
| video only      | 0.681    | 0.742 | 0.221             |
| audio only      | 0.612    | 0.658 | 0.104             |
| early fusion    | 0.703    | 0.771 | 0.283             |
| late fusion     | 0.694    | 0.759 | 0.257             |
| cross-attention | 0.784    | 0.856 | 0.441             |

Cross-attention leads by roughly 8 points of accuracy over the next best. The
late fusion gap is the informative one: it is close to early fusion on clean
clips, and it is the strategy that structurally cannot represent cross-modal
agreement. The next table shows where that matters.

## By generator family, cross-attention vs late fusion

| held-out generator | failure mode        | cross-attn | late  | gap    |
|--------------------|---------------------|------------|-------|--------|
| gen_A              | visual artifacts    | 0.812      | 0.798 | +0.014 |
| gen_B              | visual artifacts    | 0.774      | 0.761 | +0.013 |
| gen_C              | voice cloning       | 0.751      | 0.729 | +0.022 |
| gen_D              | lip-sync mismatch   | 0.803      | 0.564 | +0.239 |
| gen_E              | lip-sync mismatch   | 0.781      | 0.542 | +0.239 |

This is the experiment. On generators whose artifacts are within a single
modality, the two strategies are within 2 points. On generators whose failure
is cross-modal, late fusion falls to near chance while cross-attention holds.

The mechanism claimed in `fusion.py` predicted exactly this pattern, and the
pattern is not explainable by capacity — late fusion has comparable parameter
count and does fine on gen_A through gen_C.

## Degradation, cross-attention

| degradation     | accuracy | vs clean |
|-----------------|----------|----------|
| clean           | 0.784    | —        |
| downscale       | 0.762    | -0.022   |
| crop_face       | 0.749    | -0.035   |
| compress_medium | 0.706    | -0.078   |
| compress_heavy  | 0.612    | -0.172   |
| temporal_crop   | 0.531    | -0.253   |

**Compression is the expected loss.** CRF 40 removes the high-frequency detail
several artifact types live in. A 17-point drop is severe and it is not a
surprise.

**Temporal crop is the finding.** Accuracy falls to 0.531, near chance, and the
reason is specific: dropping video frames without dropping matching audio
induces desync in REAL clips. The model's strongest signal is desync, so it
reads legitimately cropped real clips as synthetic.

This is a false positive mode created by an entirely benign edit, and it is the
kind of thing that reaches production undetected because nobody tests the
transformation that mimics their own signal. Mitigation — alignment
verification as a preprocessing step, or desync augmentation on real clips
during training — is future work, and it is named here rather than omitted.

## The number to quote

**0.784 accuracy, 0.441 recall at 1% FPR, on unseen generators, clean clips.**

Under realistic compression, 0.706. Under temporal cropping, the model is not
usable without the mitigation above.
