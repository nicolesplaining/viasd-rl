# Convert a HuggingFace Qwen2.5 checkpoint -> gpt-fast model.pth for engine/model.py.
# Adapted from gpt-fast scripts/convert_hf_checkpoint.py (commit in GPTFAST_COMMIT.txt), plus:
#  - qkv BIASES fused into wqkv.bias, with the rotary permute applied to q,k biases
#    (gpt-fast uses interleaved-pair rope; HF uses half-split — weights AND biases must
#    both be permuted, or logits are silently garbage)
#  - asserts our ModelArgs against the HF config.json (no trusted-from-memory constants)
#  - tied-embedding models (0.5B): checkpoint has no lm_head; tiers.py assigns at load
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file

sys.path.append(str(Path(__file__).parent))
from model import ModelArgs


@torch.inference_mode()
def convert(checkpoint_dir: Path, model_name: str = None):
    if model_name is None:
        model_name = checkpoint_dir.name
    config = ModelArgs.from_name(model_name)

    # --- assert against HF config.json (catches wrong constants immediately) ---
    hf = json.load(open(checkpoint_dir / "config.json"))
    checks = {
        "hidden_size": config.dim,
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "num_key_value_heads": config.n_local_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "rope_theta": config.rope_base,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    for k, ours in checks.items():
        theirs = hf.get(k)
        assert theirs == ours, f"config mismatch {k}: HF={theirs} ours={ours}"
    print(f"[config] {model_name}: all {len(checks)} fields match HF config.json")

    # --- locate weight shards ---
    idx_st = checkpoint_dir / "model.safetensors.index.json"
    single_st = checkpoint_dir / "model.safetensors"
    if idx_st.is_file():
        bin_index = json.load(open(idx_st))
        bin_files = {checkpoint_dir / b for b in bin_index["weight_map"].values()}
    elif single_st.is_file():
        bin_files = {single_st}
    else:
        raise FileNotFoundError(f"no safetensors found under {checkpoint_dir}")

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    def permute_w(w, n_head):
        # HF half-split rope -> gpt-fast interleaved-pair rope (weights)
        return (w.view(n_head, 2, config.head_dim // 2, config.dim)
                 .transpose(1, 2)
                 .reshape(config.head_dim * n_head, config.dim))

    def permute_b(b, n_head):
        # same permutation on the bias vector
        return (b.view(n_head, 2, config.head_dim // 2)
                 .transpose(1, 2)
                 .reshape(n_head * config.head_dim))

    merged = {}
    for f in sorted(bin_files):
        merged.update(load_safetensors_file(str(f), device="cpu"))

    final = {}
    for key, value in merged.items():
        if "layers" in key:
            abstract = re.sub(r"(\d+)", "{}", key)
            ln = re.search(r"\d+", key).group(0)
            nk = weight_map[abstract]
            if nk is None:
                continue
            nk = nk.format(ln)
        else:
            nk = weight_map[key]
        final[nk] = value

    # fuse q,k,v -> wqkv (weights and biases), applying the rope permute to q and k
    for key in tuple(final.keys()):
        if key.endswith("wq.weight"):
            q = permute_w(final.pop(key), config.n_head)
            k = permute_w(final.pop(key.replace("wq.", "wk.")), config.n_local_heads)
            v = final.pop(key.replace("wq.", "wv."))
            final[key.replace("wq.weight", "wqkv.weight")] = torch.cat([q, k, v])
        elif key.endswith("wq.bias"):
            qb = permute_b(final.pop(key), config.n_head)
            kb = permute_b(final.pop(key.replace("wq.", "wk.")), config.n_local_heads)
            vb = final.pop(key.replace("wq.", "wv."))
            final[key.replace("wq.bias", "wqkv.bias")] = torch.cat([qb, kb, vb])

    if config.tie_word_embeddings and "output.weight" not in final:
        print("[tied] no lm_head in checkpoint (expected); output.weight assigned at load")

    out = checkpoint_dir / "model.pth"
    torch.save(final, out)
    print(f"[saved] {out}  ({len(final)} tensors)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=Path, required=True)
    ap.add_argument("--model_name", type=str, default=None)
    a = ap.parse_args()
    convert(a.checkpoint_dir, a.model_name)
