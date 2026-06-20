#!/usr/bin/env python3
"""Duplicate slides.tex -> slides_5min.tex, replacing each \\note{} with a condensed
(~5 min total) script. Slides themselves are unchanged."""

NOTES = [
# 1 title
"We're Nicole and Harsh, presenting S-four-D, a new state-of-the-art method for hierarchical "
"speculative decoding.",
# 2 overview
"Speculative decoding speeds up LLMs. VIA-SD adds a slim middle verifier but routes with hand-tuned "
"thresholds. Our three contributions: a learned per-token gate, a quality-versus-cost reward, and "
"per-token RL. The payoff: lossless accuracy at four times fewer calls, up to three and a half times "
"faster, a new state of the art.",
# 3 bottleneck
"Each token streams the whole model from memory, so inference is slow. Speculative decoding lets a "
"small drafter guess and the big model verify in parallel: lossless, about 1.4 times faster. The lever "
"is the acceptance rate.",
# 4 via-sd
"VIA-SD adds a slim middle verifier, the target with half its layers skipped, giving three choices: "
"accept, regenerate, or escalate. Borderline tokens go to the cheap tier. Because the model verifies "
"its own draft through a reduced copy of itself, that is the Semi-Self in our name.",
# 5 thresholds
"But VIA-SD picks between tiers with two hand-tuned thresholds, 0.5 and 0.3. They're global, brittle, "
"and cap at 0.50 accuracy even fully swept, because one cutoff can't adapt to context. Routing per "
"token is a decision problem, so the gate teaches itself: the Self-Taught.",
# 6 contribution 1
"Contribution one: the learned gate, framed as an MDP. State is cheap q-free features, action is the "
"tier, policy is a tiny MLP. The full model is used only for rewards at training time, never to decide, "
"so the gate is free to deploy.",
# 7 contribution 2
"Contribution two: the reward, match minus lambda times cost. Quality is whether our token matches the "
"full model; lambda trades accuracy for speed. One caveat: match is a proxy, and ninety-eight percent "
"token match still gives only 0.72 final accuracy, because slips compound.",
# 8 contribution 3
"Contribution three: per-token RL. Same policy gradient; they differ only in the advantage. REINFORCE "
"uses one noisy trajectory reward, GRPO normalizes within a group, and our per-token version gives each "
"decision its own advantage: lowest variance, best results. Credit assignment is everything.",
# 9 training curves
"All three converge, match reaches about 0.95, rejection falls, and per-token in blue is smoothest. "
"Match plateaus at 0.95, not 1.0, and that residual is what compounds into the accuracy gap.",
# 10 results
"The numbers. The hand-tuned thresholds, in red, are broken at 0.34 to 0.40: the slim verifier is "
"miscalibrated, so a global cutoff can't tell confident-correct from confident-wrong, and errors "
"compound. Our learned gates, in blue, recover full lossless accuracy, 0.907. This is state-of-the-art "
"hierarchical speculative decoding.",
# 11 bar
"Latency. The hand-tuned gate is slow because it over-escalates. Our learned gates are fastest, "
"with per-token on top.",
# 12 pareto
"Accuracy versus speed. Red is dominated on both axes, because one cutoff can't both avoid "
"over-escalating and avoid over-accepting. Blue, our learned gates, dominate, with a full frontier from "
"lambda. We Pareto-dominate the baseline, which makes S4D state of the art.",
# 13 ablation 1
"Which model to scale? A bigger drafter, 3B, holds 0.80 accuracy at thirty-six times fewer calls, "
"because it's accepted more. A bigger verifier, 32B, collapses to 0.33, because the weak drafter's "
"tokens no longer match. The lever is the drafter.",
# 14 ablation 2
"Two more knobs. A better slim verifier via DIMR lifts REINFORCE from 0.88 to 0.91, but is moot once "
"the gate is accept-heavy. And the honest negative: optimizing correctness directly degraded every "
"policy, so the match-trained gate stays best.",
# 15 impact
"Impact: lossless accuracy at four times fewer calls with zero accuracy lost; a tunable frontier up to "
"ten times fewer calls; and we Pareto-dominate VIA-SD at twice its speedup and higher accuracy. S4D is "
"the new state of the art in hierarchical speculative decoding.",
# 16 future
"Next: spec-spec decoding, making the drafter itself speculative; KnapSpec, spending a verifier budget "
"like a knapsack on the highest-impact tokens; a correctness-floor objective; and real wall-clock at "
"scale.",
# 17 closing
"To sum up: S4D turns hand-tuned thresholds into a learned policy and sets the state of the art in "
"hierarchical speculative decoding. Thank you.",
]

src = open("slides.tex").read()
out, i, idx = [], 0, 0
while True:
    j = src.find("\\note{", i)
    if j == -1:
        out.append(src[i:]); break
    out.append(src[i:j])
    k = j + len("\\note{"); depth = 1
    while depth > 0:
        c = src[k]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        k += 1
    out.append("\\note{\n" + NOTES[idx] + "\n}")
    idx += 1; i = k
open("slides_5min.tex", "w").write("".join(out))
wc = sum(len(n.split()) for n in NOTES)
print(f"replaced {idx} notes; total script words = {wc} (~{wc/150:.1f} min at 150 wpm)")
