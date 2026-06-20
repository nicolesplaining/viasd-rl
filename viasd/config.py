import os
from dataclasses import dataclass, field
import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # models (drafter p, verifier q; q' is q with layers skipped)
    drafter_name: str = field(default_factory=lambda: os.environ.get("VIASD_DRAFTER", "Qwen/Qwen2.5-0.5B-Instruct"))
    verifier_name: str = field(default_factory=lambda: os.environ.get("VIASD_VERIFIER", "Qwen/Qwen2.5-14B-Instruct"))
    device: str = field(default_factory=_default_device)
    dtype: str = "bfloat16"  # used on cuda; cpu/mps fall back to float32

    # decoding
    gamma: int = 5             # draft block length
    max_new_tokens: int = 320

    # slim-verifier construction (q')
    skip_ratio: float = 0.45   # fraction of layers to skip when forming q'
    keep_first_last: bool = True
    keep_mask_path: str = ""   # optional JSON from DIMR search; overrides skip_ratio

    # fixed-threshold VIA-SD baseline.
    # NOTE on the paper's notation: Algorithm 1 / Appendix B state the gates with
    # (1-alpha1), (1-alpha2) but the reported values alpha1=0.5 > alpha2=0.3 make the
    # three regions degenerate (the "regenerate" band collapses). We therefore use an
    # explicit, self-consistent convention on the confidence ratio r = q'(v)/max_u q'(u):
    #   r >= theta_accept            -> accept the drafter token   (region A)
    #   theta_escalate <= r < accept -> regenerate with q'         (region B, "middle zone")
    #   r <  theta_escalate          -> escalate to full q         (region C)
    # with theta_accept > theta_escalate. (theta_accept = 1-alpha1, theta_escalate = 1-alpha2
    # under the corrected ordering alpha1 < alpha2.)
    theta_accept: float = 0.60
    theta_escalate: float = 0.30

    seed: int = 0

    def torch_dtype(self):
        if self.device == "cuda":
            return getattr(torch, self.dtype)
        return torch.float32
