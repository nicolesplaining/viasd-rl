# Build the three tiers on the gpt-fast engine: drafter p (separate 0.5B), full verifier q,
# and slim verifier q' = a second Transformer whose parameters ARE q's kept-layer tensors
# (meta-construct + load_state_dict(assign=True) -> shared storage, zero weight duplication)
# with its OWN KV caches (required: q''s KV differs numerically -- its residual stream
# skipped layers upstream).
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from model import ModelArgs, Transformer

COMMON_VOCAB = 151936  # min(drafter, verifier) vocab; all argmax/gating on this slice


def load_engine_model(model_name: str, ckpt_dir: Path, device="cuda",
                      dtype=torch.bfloat16) -> Transformer:
    cfg = ModelArgs.from_name(model_name)
    with torch.device("meta"):                      # no fp32 CPU allocation
        model = Transformer(cfg)
    sd = torch.load(Path(ckpt_dir) / "model.pth", mmap=True, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    allowed_missing = {"output.weight"} if cfg.tie_word_embeddings else set()
    assert set(missing) <= allowed_missing, f"missing keys: {missing[:5]}"
    if cfg.tie_word_embeddings:
        model.output.weight = model.tok_embeddings.weight   # tie BEFORE .to (memo keeps it shared)
    model = model.to(device=device, dtype=dtype)
    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"param never loaded: {n}"
    return model.eval()


def build_qprime(q: Transformer, keep_mask) -> Transformer:
    """Slim-verifier view sharing q's parameter tensors. Call AFTER q is on-device.
    Never call .to() on the result (it would copy and break sharing)."""
    kept = [i for i, k in enumerate(keep_mask) if k]
    with torch.device("meta"):
        qp = Transformer(replace(q.config, n_layer=len(kept)))
    qp.tok_embeddings.weight = q.tok_embeddings.weight
    for j, i in enumerate(kept):
        # parameters ONLY -- kv_cache buffers must never be shared between q and q'
        sd = {k: v for k, v in q.layers[i].state_dict().items() if "kv_cache" not in k}
        qp.layers[j].load_state_dict(sd, assign=True)
    qp.norm.weight = q.norm.weight
    qp.output.weight = q.output.weight
    for n, p in qp.named_parameters():
        assert p.device.type != "meta", f"unshared param {n}"
        assert p.device == q.output.weight.device
    return qp.eval()


def setup_all_caches(models, max_seq: int, device="cuda"):
    with torch.device(device):
        for m in models:
            m.setup_caches(1, max_seq)


def load_keep_mask(path) -> list:
    return json.load(open(path))["keep_mask"]


@dataclass
class Engine:
    drafter: Transformer
    q: Transformer
    qp: Transformer            # None for plain-SD-only setups
    vocab: int
    gamma: int
    max_seq: int
    device: str
    # compiled callables (one compiled object per model; shapes specialize + cudagraph per shape)
    drafter_fwd: object = None
    q_fwd: object = None
    qp_fwd: object = None

    def compile_all(self):
        import torch._inductor.config as ic
        ic.coordinate_descent_tuning = True   # gpt-fast's batch-1 GEMV magic: Triton beats
        ic.triton.unique_kernel_names = True  # cuBLAS ~1.5x on skinny matmuls
        def make(m):
            def fwd(x, pos):
                return m(x, pos)
            return torch.compile(fwd, mode="reduce-overhead", fullgraph=True)
        self.drafter_fwd = make(self.drafter)
        self.q_fwd = make(self.q)
        if self.qp is not None:
            self.qp_fwd = make(self.qp)
        return self

    def eager(self):
        self.drafter_fwd = lambda x, pos: self.drafter(x, pos)
        self.q_fwd = lambda x, pos: self.q(x, pos)
        if self.qp is not None:
            self.qp_fwd = lambda x, pos: self.qp(x, pos)
        return self


def build_engine(drafter_dir, verifier_dir, drafter_name=None, verifier_name=None,
                 keep_mask=None, self_draft_mask=None, gamma=5, max_seq=1408, device="cuda",
                 dtype=torch.bfloat16, compile=True) -> Engine:
    q = load_engine_model(verifier_name or Path(verifier_dir).name, verifier_dir, device, dtype)
    if self_draft_mask is not None:
        # FULL-SELF (S5D): the drafter is a layer-subset VIEW of the verifier itself --
        # one weight set plays all three roles (draft / q' / q).
        drafter = build_qprime(q, self_draft_mask)
    else:
        drafter = load_engine_model(drafter_name or Path(drafter_dir).name, drafter_dir, device, dtype)
    qp = build_qprime(q, keep_mask) if keep_mask is not None else None
    models = [m for m in (drafter, q, qp) if m is not None]
    setup_all_caches(models, max_seq, device)
    eng = Engine(drafter=drafter, q=q, qp=qp, vocab=COMMON_VOCAB, gamma=gamma,
                 max_seq=max_seq, device=device)
    return eng.compile_all() if compile else eng.eager()
