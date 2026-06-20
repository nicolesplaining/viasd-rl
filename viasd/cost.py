"""Cost / latency accounting.

Speculative decoding trades *extra FLOPs* for *lower latency* (it verifies tokens
it then rejects). So we do NOT report a FLOP count. Instead we count, per run, the
deployment-relevant *forward calls* of each tier, and convert them to an estimated
wall-clock latency using per-forward times measured once on the actual GPU
(see models.measure_latencies). A single verifier forward over a gamma-token block
is one parallel step regardless of gamma, which is exactly why SD is faster.
"""
from dataclasses import dataclass


@dataclass
class CostMeter:
    draft_steps: int = 0    # single-token drafter forwards (sequential)
    qp_forwards: int = 0    # slim-verifier q' parallel block forwards
    q_forwards: int = 0     # full-verifier q parallel block forwards (the expensive tier)
    q1_steps: int = 0       # full-verifier 1-token cached steps (greedy-q baseline only)
    tokens: int = 0         # tokens emitted

    drafted: int = 0        # drafted tokens that were gated
    changed: int = 0        # drafted tokens not kept as-is (rejections)
    accept: int = 0         # action counts
    regen: int = 0
    escalate: int = 0

    def merge(self, o: "CostMeter") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(o, f))

    # ---- derived metrics ----
    @property
    def rejection_rate(self) -> float:
        return self.changed / self.drafted if self.drafted else 0.0

    @property
    def q_calls_per_token(self) -> float:
        # full-verifier invocations per token (a block verify counts as one invocation)
        return (self.q_forwards + self.q1_steps) / self.tokens if self.tokens else 0.0

    def tier_fractions(self):
        tot = self.accept + self.regen + self.escalate
        if not tot:
            return (0.0, 0.0, 0.0)
        return (self.accept / tot, self.regen / tot, self.escalate / tot)


@dataclass
class Latencies:
    """Seconds per forward, measured on the target GPU."""
    t_p1: float   # drafter, 1 new token (cached)
    t_qp: float   # q', gamma-token parallel block forward
    t_q: float    # q,  gamma-token parallel block forward
    t_q1: float   # q,  1 new token (cached) -- baseline autoregressive step

    def estimate(self, m: CostMeter) -> float:
        return (m.draft_steps * self.t_p1 + m.qp_forwards * self.t_qp
                + m.q_forwards * self.t_q + m.q1_steps * self.t_q1)

    def baseline(self, m: CostMeter) -> float:
        # greedy autoregressive with the full verifier
        return m.tokens * self.t_q1

    def speedup(self, m: CostMeter) -> float:
        est = self.estimate(m)
        return self.baseline(m) / est if est > 0 else 0.0
