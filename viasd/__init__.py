"""VIA-SD with a learned per-token gating policy.

Reimplementation of the hierarchical draft / slim-verify / full-verify pipeline
from "VIA-SD: Verification via Intra-Model Routing for Speculative Decoding"
(arXiv:2606.12243), extended with an RL-trained per-token gating policy that
replaces the paper's fixed (alpha1, alpha2) confidence thresholds.
"""
