#!/usr/bin/env python3
"""
gemma3_probe.py - structural inspector for Gemma 3 checkpoints.

DESIGN NOTE (read this first)
-----------------------------
This script *discovers* structure instead of assuming it. HF's Gemma 3
module layout and config field names have moved across transformers
releases:

  * 1B / text-only checkpoints load as Gemma3ForCausalLM  (flat layout)
  * 4B+ multimodal load as Gemma3ForConditionalGeneration (nested under
    a language_model / vision_tower split)
  * the local-vs-global layer flag has appeared as config.layer_types,
    as config.sliding_window_pattern, and as a per-module .is_sliding

So every lookup below probes several candidate names and prints which one
it actually found. An "UNKNOWN" in the output is information, not a bug:
it tells you your transformers version exposes that fact somewhere else.
Grep the printed module tree and add the name to the candidate list.

STATUS: exercised end to end against HF's real Gemma3ForCausalLM class
using a randomly-initialised model. All five sections run. It has still
NOT been run against real Gemma 3 weights. Random weights validate the
plumbing, not the model - treat the KV numbers as things to verify on
first run, not as established output. See README for the four bugs this
testing found.

USAGE
-----
    # config-only sections, no weight download
    python gemma3_probe.py --model google/gemma-3-1b-it \
        --sections config,kv --no-weights

    # full inspection incl. a forward pass
    python gemma3_probe.py --model google/gemma-3-1b-it --sections all
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict, defaultdict


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _dtype_kwarg(dt):
    """transformers v5 renamed from_pretrained(torch_dtype=) to (dtype=)."""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dt} if major >= 5 else {"torch_dtype": dt}


def probe(obj, *names, default="UNKNOWN"):
    """Return the first attribute in `names` that exists on `obj`.

    Returns (value, name_that_worked). Keeps the provenance so the output
    tells you which schema your transformers version is using.
    """
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v, n
    return default, None


def table(rows, headers):
    """Minimal fixed-width table printer (no pandas dependency)."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.2f}T"


def human_bytes(n):
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


# --------------------------------------------------------------------------
# 1. config resolution
# --------------------------------------------------------------------------

def resolve_text_config(cfg):
    """Gemma 3 multimodal configs nest the decoder under .text_config."""
    return getattr(cfg, "text_config", cfg)


def section_config(cfg):
    tc = resolve_text_config(cfg)
    print("\n=== CONFIG ===")
    print(f"architectures      : {getattr(cfg, 'architectures', 'UNKNOWN')}")
    print(f"nested text_config : {tc is not cfg}")
    print(f"has vision_config  : {hasattr(cfg, 'vision_config')}")

    fields = [
        ("hidden_size",         ("hidden_size",)),
        ("num_hidden_layers",   ("num_hidden_layers", "num_layers")),
        ("num_attention_heads", ("num_attention_heads",)),
        ("num_key_value_heads", ("num_key_value_heads",)),
        ("head_dim",            ("head_dim",)),
        ("intermediate_size",   ("intermediate_size",)),
        ("vocab_size",          ("vocab_size",)),
        ("max_position_emb",    ("max_position_embeddings",)),
        ("rope_theta (global)", ("rope_theta",)),
        ("rope_local_base_freq",("rope_local_base_freq",)),
        ("sliding_window",      ("sliding_window",)),
        ("sliding_window_patt", ("sliding_window_pattern",)),
        ("layer_types",         ("layer_types",)),
        ("tie_word_embeddings", ("tie_word_embeddings",)),
        ("hidden_activation",   ("hidden_activation", "hidden_act")),
        ("query_pre_attn_scalar", ("query_pre_attn_scalar",)),
    ]
    rows = []
    for label, names in fields:
        val, found = probe(tc, *names)
        if isinstance(val, list) and len(val) > 8:
            val = f"[{val[0]}, {val[1]}, ... ] len={len(val)}"
        rows.append([label, val, found or "-"])
    table(rows, ["field", "value", "cfg attr"])

    # head_dim is often absent and must be derived
    hd, _ = probe(tc, "head_dim")
    if hd == "UNKNOWN":
        h, _ = probe(tc, "hidden_size")
        n, _ = probe(tc, "num_attention_heads")
        if "UNKNOWN" not in (h, n):
            print(f"\nnote: head_dim absent from config; hidden/heads = {h // n}")
            print("      (Gemma 3 does NOT always satisfy head_dim*n_heads == hidden_size,")
            print("       so prefer the explicit field or a shape read off q_proj.)")
    return tc


# --------------------------------------------------------------------------
# 2. module discovery + local/global classification
# --------------------------------------------------------------------------

def find_decoder_layers(model):
    """Locate the ModuleList of decoder blocks without hardcoding a path.

    Looks for the first ModuleList whose children expose both `self_attn`
    and `mlp`. Survives the flat-vs-nested layout difference.
    """
    import torch.nn as nn
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            child = mod[0]
            if hasattr(child, "self_attn") and hasattr(child, "mlp"):
                return name, mod
    raise RuntimeError("no decoder ModuleList found; print model and adjust")


def classify_layers(model, cfg):
    """Return [(idx, kind, window, rope_base, signal_source), ...]."""
    tc = resolve_text_config(cfg)
    path, layers = find_decoder_layers(model)
    print(f"\ndecoder layers found at: {path}  (n={len(layers)})")

    layer_types, _ = probe(tc, "layer_types")
    pattern, _ = probe(tc, "sliding_window_pattern")
    window, _ = probe(tc, "sliding_window")

    out = []
    for i, blk in enumerate(layers):
        attn = blk.self_attn
        kind, src = None, None

        # signal 1: per-module attribute
        for a in ("is_sliding", "attention_type", "layer_type"):
            if hasattr(attn, a):
                v = getattr(attn, a)
                kind = ("local" if v else "global") if isinstance(v, bool) \
                    else ("local" if "slid" in str(v) else "global")
                src = f"module.{a}"
                break

        # signal 2: config.layer_types list
        if kind is None and isinstance(layer_types, list) and i < len(layer_types):
            kind = "local" if "slid" in str(layer_types[i]) else "global"
            src = "cfg.layer_types"

        # signal 3: derive from the 5:1 pattern
        if kind is None and isinstance(pattern, int):
            kind = "local" if bool((i + 1) % pattern) else "global"
            src = "cfg.sliding_window_pattern"

        if kind is None:
            kind, src = "UNKNOWN", "-"

        w = window if kind == "local" else "full"
        rope, _ = probe(attn, "rope_theta", "base", default="-")
        out.append((i, kind, w, rope, src))

    return out


def section_layers(model, cfg):
    print("\n=== LAYERS ===")
    rows = classify_layers(model, cfg)
    table(rows, ["idx", "kind", "kv window", "rope base", "signal"])

    kinds = [r[1] for r in rows]
    print(f"\nlocal={kinds.count('local')}  global={kinds.count('global')}  "
          f"unknown={kinds.count('UNKNOWN')}")
    print("expected for Gemma 3: 5 local per 1 global, layer 0 local.")
    print("If you see something else, trust the module tree over this comment.")
    return rows


# --------------------------------------------------------------------------
# 3. parameter accounting
# --------------------------------------------------------------------------

BUCKETS = OrderedDict([
    ("embeddings",     r"embed_tokens"),
    ("attn.q_proj",    r"self_attn\.q_proj"),
    ("attn.k_proj",    r"self_attn\.k_proj"),
    ("attn.v_proj",    r"self_attn\.v_proj"),
    ("attn.o_proj",    r"self_attn\.o_proj"),
    ("attn.qk_norm",   r"self_attn\.[qk]_norm"),
    ("mlp.gate_proj",  r"mlp\.gate_proj"),
    ("mlp.up_proj",    r"mlp\.up_proj"),
    ("mlp.down_proj",  r"mlp\.down_proj"),
    ("norms",          r"(layernorm|_norm)$|norm\.weight"),
    ("vision",         r"vision_tower|vision_model|multi_modal_projector"),
])


def section_params(model):
    print("\n=== PARAMETERS ===")
    counts = defaultdict(int)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        for bucket, pat in BUCKETS.items():
            if re.search(pat, name):
                counts[bucket] += n
                break
        else:
            counts["other"] += n

    rows = [[b, human(c), f"{100 * c / total:5.1f}%"]
            for b, c in sorted(counts.items(), key=lambda kv: -kv[1])]
    table(rows, ["component", "params", "share"])
    print(f"\ntotal: {human(total)}")
    print("Read this as: attention projections are cheap, MLP is where the")
    print("parameters live, and the 256k-entry vocab makes embeddings a")
    print("large fraction at small model sizes.")


# --------------------------------------------------------------------------
# 4. KV-cache arithmetic  (the point of the 5:1 interleave)
# --------------------------------------------------------------------------

def section_kv(cfg, layer_rows=None, seq_len=32768, dtype_bytes=2):
    print(f"\n=== KV CACHE @ seq_len={seq_len} ===")
    tc = resolve_text_config(cfg)
    n_kv, _ = probe(tc, "num_key_value_heads")
    hd, _ = probe(tc, "head_dim")
    n_layers, _ = probe(tc, "num_hidden_layers")
    window, _ = probe(tc, "sliding_window")
    pattern, _ = probe(tc, "sliding_window_pattern")

    if "UNKNOWN" in (n_kv, hd, n_layers):
        print("missing config fields; run the config section and fill in manually")
        return

    if layer_rows:
        kinds = [r[1] for r in layer_rows]
    elif isinstance(pattern, int):
        kinds = ["local" if bool((i + 1) % pattern) else "global"
                 for i in range(n_layers)]
    else:
        print("cannot determine layer kinds; assuming all-global")
        kinds = ["global"] * n_layers

    per_pos = 2 * n_kv * hd * dtype_bytes          # K and V
    local_pos = min(seq_len, window) if isinstance(window, int) else seq_len

    actual = sum(per_pos * (local_pos if k == "local" else seq_len) for k in kinds)
    all_global = per_pos * seq_len * n_layers

    table([
        ["bytes / position / layer", human_bytes(per_pos)],
        ["local layers", f"{kinds.count('local')} x {local_pos} pos"],
        ["global layers", f"{kinds.count('global')} x {seq_len} pos"],
        ["actual cache", human_bytes(actual)],
        ["hypothetical all-global", human_bytes(all_global)],
        ["reduction", f"{100 * (1 - actual / all_global):.1f}%"],
    ], ["quantity", "value"])
    print("\nThis ratio is the entire argument for the interleave. Re-run at")
    print("seq_len=1024 and seq_len=131072 and watch it change.")


# --------------------------------------------------------------------------
# 5. live activation capture
# --------------------------------------------------------------------------

def section_hooks(model, tokenizer, text, capture_attn=False):
    import torch

    print("\n=== ACTIVATIONS ===")
    _, layers = find_decoder_layers(model)
    store = {}
    handles = []

    def mk(tag):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if hasattr(t, "shape"):
                store[tag] = (tuple(t.shape), float(t.float().norm()))
        return hook

    for i, blk in enumerate(layers):
        handles.append(blk.register_forward_hook(mk(f"L{i:02d}.residual")))
        handles.append(blk.self_attn.register_forward_hook(mk(f"L{i:02d}.attn_out")))
        handles.append(blk.mlp.register_forward_hook(mk(f"L{i:02d}.mlp_out")))

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs,
                    output_hidden_states=True,
                    output_attentions=capture_attn)

    for h in handles:
        h.remove()

    rows = [[k, str(v[0]), f"{v[1]:.1f}"] for k, v in sorted(store.items())]
    table(rows, ["site", "shape", "L2 norm"])

    a = getattr(out, "attentions", None)
    # sdpa returns a TUPLE OF NONES, which is truthy. Check the element.
    if capture_attn and a and a[0] is not None:
        print(f"\nattention tensors: {len(a)} layers, shape[0]={tuple(a[0].shape)}")
        print("(batch, heads, q_len, kv_len) - heatmap these per layer.")
        print("If this is None, you did not load with attn_implementation='eager'.")
    elif capture_attn:
        print("\nattentions came back empty -> attn_implementation is not 'eager'")

    print("\nPlot residual L2 norm vs layer index. Gemma 3 uses pre- AND")
    print("post-norm around each sublayer, so the growth curve should look")
    print("different from a pre-norm-only model like Llama.")
    return store


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--sections", default="all",
                    help="comma list of: config,layers,params,kv,hooks | all")
    ap.add_argument("--no-weights", action="store_true",
                    help="config-only; works for config and kv sections")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--attn", action="store_true",
                    help="also capture attention matrices (forces eager)")
    args = ap.parse_args()

    want = ({"config", "layers", "params", "kv", "hooks"}
            if args.sections == "all" else set(args.sections.split(",")))

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)

    tc = section_config(cfg) if "config" in want else resolve_text_config(cfg)

    model = tokenizer = None
    layer_rows = None

    if not args.no_weights and want & {"layers", "params", "hooks"}:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = _dtype_kwarg(getattr(torch, args.dtype))
        if args.attn:
            kw["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if "layers" in want and model is not None:
        layer_rows = section_layers(model, cfg)
    if "params" in want and model is not None:
        section_params(model)
    if "kv" in want:
        section_kv(cfg, layer_rows, seq_len=args.seq_len)
    if "hooks" in want and model is not None:
        section_hooks(model, tokenizer, args.text, capture_attn=args.attn)


if __name__ == "__main__":
    main()
