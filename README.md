# Google DeepMind | Audio-Visual Fusion — Build the Eval Before the Model

An intermediate multimodal ML project shaped like the Human Understanding research engineering problem — deciding whether a talking-head clip is a real person or a synthetic impersonation, using audio and video together. The model is the easy half. The hard half, and the one this project is really about, is the evaluation: you build a held-out-generator split before you train anything, because accuracy measured on generators the model has already seen is close to meaningless and will be high enough to fool you. You implement three fusion strategies and find that the one which scores best in-distribution is not the one that generalizes, run an adversarial degradation suite that a naive detector fails completely, and produce an ablation table that says honestly how much each modality contributed. The deliverable is a number you are allowed to believe.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- Python
- PyTorch
- NumPy
- scikit-learn
- ffmpeg
- pytest
