# 모델·참조 코드 출처

- ProbPose-s weights: `vrg-prague/ProbPose-s`, HF model card GPL-3.0.
  SHA-256: `54c4e8cf64ea58c38f7ef49d19e7c0cd6cf6e4de567e23dc1c0ab607122e16f6`.
- ProbPose reference code: https://github.com/paingoat/ProbPose
  revision `1b20a3000e461d795625af02836a00e0511c23c3`.
  `third_party/probpose_post_processing.py`, `reference_probmap_head.py`,
  `reference_probpose_config.py` are retained upstream files. Original copyright
  headers and repository license are retained in `third_party/ProbPose-LICENSE`.
  The portable inference head is adapted from ProbMapHead, not an independently
  trained model. Model weights have separate terms from repository code.
- mmpretrain/MMCV/MMEngine, RTMW, YOLOX, rtmlib: OpenMMLab/Tau-J official software
  and checkpoints. RTMW source URL and detector source URL are recorded in each
  run. See respective repositories for code/model terms.
- ONNX Runtime, Streamlit, PyTorch and other packages: respective package licenses.

This prototype does not assign a new license to third-party weights or code.

- Google DeepMind TAP source: https://github.com/google-deepmind/tapnet
  pinned revision `730cda1c730877cfedbe01bf87fb1cadb78a565d`.
  Official PyTorch `Online BootsTAPIR` architecture and weights.
  Source and linked model weights: Apache-2.0 per upstream README.
  Weights: https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt
  SHA-256: `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.
  Source is kept unmodified under `vendor/tapnet`. `scripts/setup_tap.py` restores
  the pinned source when needed; `third_party/TAP-LICENSE` preserves the license.
  Core ML export-only adaptations are documented in `posecheck/tap_export.py`.
