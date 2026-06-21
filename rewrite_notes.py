#!/usr/bin/env python3
"""Replace the 10 \\note{} blocks in slides.tex with a formal, academic conference-talk script."""
SS = r"S\textsuperscript{4}D"

NOTES = [
# 1 title
f"Thank you. We present {SS}, Self-Taught Semi-Self Speculative Decoding, a reinforcement-learned "
"routing policy for hierarchical speculative decoding. We will show that it establishes a new state "
"of the art on this problem.",
# 2 background
"We begin with the setting. Speculative decoding accelerates language-model inference without altering "
"the output distribution: a small drafter proposes a block of tokens, and the target model verifies "
"the entire block in a single parallel pass, accepting the longest prefix consistent with its own "
"distribution. Its efficiency is therefore governed by the acceptance rate. VIA-SD extends this scheme "
"with an intermediate verifier, a layer-skipped copy of the target, and introduces a three-way routing "
"decision per token: accept the draft, regenerate with the slim verifier, or escalate to the full "
"model. However, VIA-SD selects among these tiers using two fixed, hand-tuned confidence thresholds. "
"This rule is global and static, and we find it to be markedly sub-optimal.",
# 3 method
"Our central idea is to treat this routing decision as a learning problem. We formulate per-token tier "
"selection as a Markov decision process. The state is a compact set of features that do not require "
"the target model; the action space is the three routing tiers; and the policy is a small multilayer "
"perceptron. The reward balances fidelity to the target model against computational cost, governed by "
"a single coefficient. We train the policy first by imitation and then by reinforcement learning. "
"Crucially, the target model is consulted only to compute rewards during training, so the gate "
"operates without it at inference time.",
# 4 rl
"We study three reinforcement-learning formulations. All three optimize the same policy-gradient "
"objective with an entropy regularizer, and differ only in how the advantage is estimated. REINFORCE "
"uses a global moving-average baseline; GRPO uses a group-relative advantage with proximal-policy "
"clipping and a Kullback-Leibler penalty toward the warm-started policy; and our per-token variant "
"assigns each decision a batch-normalized advantage.",
# 5 results
"We evaluate on GSM8K with a Qwen2.5 drafter and verifier. The learned gate attains between 0.88 and "
"0.91 accuracy, whereas the hand-tuned thresholds reach only 0.21 to 0.40. REINFORCE combined with our "
"learned slim verifier recovers lossless accuracy, and GRPO achieves the fewest full-model calls. "
"Measured in full-model calls per token, a hardware-independent metric, the learned policies are "
"uniformly superior.",
# 6 pareto
"These results define a frontier in the accuracy-efficiency plane. The learned policies Pareto-dominate "
"the fixed-threshold baseline, and the cost coefficient traces a continuous trade-off between accuracy "
"and speed. Strengthening the drafter shifts this frontier outward. To our knowledge, this constitutes "
"the state of the art for hierarchical speculative decoding.",
# 7 conclusion
f"To conclude, {SS} replaces VIA-SD's hand-tuned thresholds with a learned, per-token routing policy. "
"It matches lossless accuracy at roughly a quarter of the full-model calls, attains up to a 3.5-fold "
"speedup, deploys as a small target-free network, and improves as drafters improve. We therefore "
"present it as a new state of the art in hierarchical speculative decoding.",
# 8 thank you
"Thank you for your attention. We would be glad to take questions.",
# 9 ablations (backup)
"We briefly summarize two ablations. Scaling the drafter to three billion parameters preserves accuracy "
"while sharply reducing full-model calls, whereas scaling the verifier to thirty-two billion parameters "
"degrades an acceptance-heavy policy. Separately, a slim verifier obtained through our layer-skip search "
"improves the balanced policies, and directly optimizing final-answer correctness did not surpass the "
"policy trained on the token-level objective.",
# 10 future (backup)
"Finally, we outline several directions: recursive speculative decoding, in which the drafter is itself "
"made speculative; a knapsack formulation that allocates a fixed budget of full-model calls across "
"tokens; a constrained objective that minimizes cost subject to an accuracy floor; and evaluation of "
"wall-clock latency on an optimized inference engine.",
]

s = open("slides.tex").read()
out, i, idx = [], 0, 0
while True:
    j = s.find("\\note{", i)
    if j == -1:
        out.append(s[i:]); break
    out.append(s[i:j])
    k, depth = j + 6, 1
    while depth > 0:
        c = s[k]; depth += (c == "{") - (c == "}"); k += 1
    out.append("\\note{\n" + NOTES[idx] + "\n}")
    idx += 1; i = k
assert idx == len(NOTES), f"{idx} notes in file vs {len(NOTES)} provided"
open("slides.tex", "w").write("".join(out))
print(f"rewrote {idx} notes; {sum(len(n.split()) for n in NOTES)} words")
