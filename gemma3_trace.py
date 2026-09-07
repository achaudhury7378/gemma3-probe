#!/usr/bin/env python3
"""
gemma3_trace.py - layers x positions trace of a Gemma 3 forward pass.

WHAT THIS IS ACTUALLY SHOWING
-----------------------------
Prefill is parallel. Every prompt position passes through every layer in
one forward pass; only the causal mask makes it "sequential". So the
object of interest is a GRID:

        positions ->
  L0    [ what layer 0 thinks the next token is, at each position ]
  L1    [ ... ]
  ...
  Ln    [ the actual prediction ]

Reading down a column shows the prediction crystallizing for one position.
Reading across a row shows what a single layer contributes everywhere.
That is the logit lens, and it is the closest honest thing to "watching
the model think".

Genuinely sequential behaviour only exists during generation, where each
new token gets its own pass. --generate covers that separately.

SELF-VERIFICATION
-----------------
The one detail that silently ruins logit lens is whether hidden_states[-1]
has already had the final RMSNorm applied. This differs by model and by
transformers version, and I did not want to assert it from memory. So the
script DETECTS it: it reconstructs the real logits both ways and uses
whichever matches. If neither matches closely, it says so loudly rather
than producing a plausible-looking wrong grid.

STATUS: exercised end to end against HF's real Gemma3ForCausalLM class
using a randomly-initialised model. All sections run, and the detection
step below correctly identified hidden_states[-1] as post-final-norm
(raw diff 0.0 vs normed diff 6e-07). It has still NOT been run against
real Gemma 3 weights.

USAGE
-----
    python gemma3_trace.py --model google/gemma-3-1b-it \
        --prompt "The capital of France is" --sections lens,delta,surprisal

    python gemma3_trace.py --model google/gemma-3-1b-it \
        --prompt "..." --sections attn --attn --target -1

    python gemma3_trace.py --model google/gemma-3-1b-it \
        --prompt "..." --generate 8
"""

from __future__ import annotations

import argparse


# --------------------------------------------------------------------------
# module location (same discovery approach as gemma3_probe.py)
# --------------------------------------------------------------------------

def _dtype_kwarg(dt):
    """transformers v5 renamed from_pretrained(torch_dtype=) to (dtype=)."""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dt} if major >= 5 else {"torch_dtype": dt}


def embed_scale(model):
    """Gemma multiplies embeddings by sqrt(hidden_size) BEFORE layer 0, but
    hidden_states[0] is the UNSCALED embedding. Verified empirically: without
    this correction the layer-0 delta reads ~8x too large."""
    import math
    cfg = model.config
    cfg = getattr(cfg, "text_config", cfg)
    return math.sqrt(cfg.hidden_size)


def scaled_hidden(model, hidden_states):
    """hidden_states with entry 0 put on the same footing as the rest."""
    hs = list(hidden_states)
    hs[0] = hs[0] * embed_scale(model)
    return hs


def get_decoder(model):
    if hasattr(model, "get_decoder"):
        try:
            d = model.get_decoder()
            if d is not None and hasattr(d, "norm"):
                return d
        except Exception:
            pass
    for name, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "norm"):
            return mod
    raise RuntimeError("decoder not found; print(model) and adjust")


def get_unembed(model):
    """Gemma ties embeddings, so this is the embedding matrix transposed."""
    oe = model.get_output_embeddings()
    if oe is None:
        raise RuntimeError("no output embeddings exposed")
    return oe.weight            # (vocab, d_model)


# --------------------------------------------------------------------------
# the critical calibration step
# --------------------------------------------------------------------------

def detect_norm_convention(model, hidden_states, logits, tol=1e-2):
    """Is hidden_states[-1] pre- or post-final-norm? Decide empirically.

    Reconstructs the real logits both ways and compares. Returns
    ('pre'|'post'|'unknown', d_raw, d_norm).
    """
    import torch
    dec = get_decoder(model)
    W_U = get_unembed(model)
    h = hidden_states[-1][0].float()

    with torch.no_grad():
        d_raw = (h @ W_U.float().T - logits[0].float()).abs().max().item()
        d_norm = (dec.norm(hidden_states[-1])[0].float() @ W_U.float().T
                  - logits[0].float()).abs().max().item()

    if min(d_raw, d_norm) > tol:
        return "unknown", d_raw, d_norm
    return ("post" if d_raw < d_norm else "pre"), d_raw, d_norm


def lens_logits(model, h, apply_norm):
    """Project one layer's residual stream to vocab space."""
    import torch
    dec = get_decoder(model)
    W_U = get_unembed(model)
    with torch.no_grad():
        x = dec.norm(h) if apply_norm else h
        return (x[0].float() @ W_U.float().T)      # (T, vocab)


# --------------------------------------------------------------------------
# section: logit lens grid
# --------------------------------------------------------------------------

def section_lens(model, tok, out, ids, convention, max_pos=12, topk=1):
    print("\n=== LOGIT LENS  (rows=layers, cols=positions) ===")
    hs = scaled_hidden(model, out.hidden_states)
    n_layers = len(hs) - 1
    T = ids.shape[1]
    cols = list(range(max(0, T - max_pos), T))

    header = "layer | " + " | ".join(
        f"{tok.decode([ids[0, p]]).strip()[:9]:>9}" for p in cols)
    print(header)
    print("-" * len(header))

    for L in range(len(hs)):
        # last entry may already be normed; all others are not
        apply_norm = not (L == len(hs) - 1 and convention == "post")
        lg = lens_logits(model, hs[L], apply_norm)   # (T, vocab)
        cells = []
        for p in cols:
            probs = lg[p].softmax(-1)
            v, i = probs.topk(topk)
            s = tok.decode([i[0].item()]).strip()[:6]
            cells.append(f"{s:>6}{v[0].item():>3.0%}")
        tag = "emb" if L == 0 else f"L{L - 1:02d}"
        print(f"{tag:>5} | " + " | ".join(f"{c:>9}" for c in cells))
        del lg

    print("\nRead a column downward: that is one position's prediction")
    print("resolving. Sharp transitions mark the layers doing the work.")
    print("Note: intermediate layers are off-distribution for the unembed,")
    print("so early-layer output is often noise, not a real 'belief'.")


# --------------------------------------------------------------------------
# section: per-layer residual delta
# --------------------------------------------------------------------------

def section_delta(model, out, tok, ids, max_pos=12):
    import torch
    print("\n=== PER-LAYER RESIDUAL CHANGE (relative delta norm) ===")
    hs = scaled_hidden(model, out.hidden_states)
    T = ids.shape[1]
    cols = list(range(max(0, T - max_pos), T))

    header = "layer | " + " | ".join(
        f"{tok.decode([ids[0, p]]).strip()[:6]:>6}" for p in cols)
    print(header)
    print("-" * len(header))

    for L in range(1, len(hs)):
        a, b = hs[L - 1][0].float(), hs[L][0].float()
        rel = ((b - a).norm(dim=-1) / (a.norm(dim=-1) + 1e-6))
        cells = " | ".join(f"{rel[p].item():>6.3f}" for p in cols)
        print(f"{f'L{L - 1:02d}':>5} | {cells}")

    print(f"\n(entry 0 rescaled by sqrt(hidden_size)={embed_scale(model):.2f};")
    print(" without that, L00 reads ~8x too large and the row is meaningless.)")
    print("High values = that layer rewrote the residual at that position.")
    print("Compare local vs global layers using the classification from")
    print("gemma3_probe.py; they should behave differently on long inputs.")


# --------------------------------------------------------------------------
# section: surprisal of the actual next token
# --------------------------------------------------------------------------

def section_surprisal(out, tok, ids):
    import torch
    print("\n=== PER-POSITION SURPRISAL (final layer) ===")
    logits = out.logits[0].float()
    logprobs = logits.log_softmax(-1)
    probs = logits.softmax(-1)
    ent = -(probs * logprobs).sum(-1)

    rows = []
    for p in range(ids.shape[1] - 1):
        nxt = ids[0, p + 1].item()
        rows.append((
            p,
            tok.decode([ids[0, p]]).strip()[:14],
            tok.decode([nxt]).strip()[:14],
            -logprobs[p, nxt].item(),
            ent[p].item(),
        ))

    print(f"{'pos':>4} {'token':>14} {'next':>14} {'surprisal':>10} {'entropy':>8}")
    print("-" * 54)
    for r in rows:
        print(f"{r[0]:>4} {r[1]:>14} {r[2]:>14} {r[3]:>10.3f} {r[4]:>8.3f}")

    print("\nSurprisal is in nats. High values are where the prompt told the")
    print("model something it could not have guessed - the informative tokens.")


# --------------------------------------------------------------------------
# section: attention flow into one position
# --------------------------------------------------------------------------

def section_attn(out, tok, ids, target=-1, max_src=12):
    print("\n=== ATTENTION INTO ONE POSITION ===")
    atts = getattr(out, "attentions", None)
    # sdpa returns a TUPLE OF NONES, which is truthy. Check the element.
    if not atts or atts[0] is None:
        print("no attention tensors returned.")
        print("re-run with --attn (loads with attn_implementation='eager').")
        return

    T = ids.shape[1]
    t = target if target >= 0 else T + target
    src = list(range(max(0, t - max_src + 1), t + 1))

    header = "layer | " + " | ".join(
        f"{tok.decode([ids[0, s]]).strip()[:6]:>6}" for s in src)
    print(f"target position {t}: {tok.decode([ids[0, t]])!r}")
    print(header)
    print("-" * len(header))

    for L, a in enumerate(atts):
        w = a[0, :, t, :].float().mean(0)     # mean over heads
        cells = " | ".join(f"{w[s].item():>6.3f}" for s in src)
        print(f"{f'L{L:02d}':>5} | {cells}")

    print("\nAveraging over heads hides specialisation - drop the .mean(0)")
    print("and print per head once a layer looks interesting.")
    print("Caveat: attention weight is not information flow. Use it to")
    print("generate hypotheses, then confirm with activation patching.")


# --------------------------------------------------------------------------
# genuinely sequential: generation
# --------------------------------------------------------------------------

def section_generate(model, tok, ids, n_new=8):
    import torch
    print(f"\n=== GENERATION TRACE ({n_new} steps) ===")
    print(f"{'step':>4} {'token':>16} {'p':>7} {'entropy':>8} {'top-3 alternatives'}")
    print("-" * 72)

    cur = ids
    for step in range(n_new):
        with torch.no_grad():
            o = model(input_ids=cur)
        lg = o.logits[0, -1].float()
        probs = lg.softmax(-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum().item()
        v, i = probs.topk(3)
        nxt = i[0].unsqueeze(0).unsqueeze(0)
        alts = "  ".join(f"{tok.decode([i[k].item()]).strip()!r}:{v[k]:.2f}"
                         for k in range(3))
        print(f"{step:>4} {tok.decode([i[0].item()]).strip()[:16]:>16} "
              f"{v[0].item():>7.3f} {ent:>8.3f} {alts}")
        cur = torch.cat([cur, nxt.to(cur.device)], dim=1)

    print("\nThis loop is the only truly sequential part. Note it recomputes")
    print("the full prefix each step - no KV cache. Fine for tracing, wrong")
    print("for benchmarking.")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--sections", default="lens,delta,surprisal")
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--target", type=int, default=-1)
    ap.add_argument("--max-pos", type=int, default=12)
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    want = set(args.sections.split(",")) if args.sections else set()

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = _dtype_kwarg(getattr(torch, args.dtype))
    if args.attn:
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids,
                    output_hidden_states=True,
                    output_attentions=args.attn)

    n_layers = len(out.hidden_states) - 1
    print(f"prompt: {args.prompt!r}")
    print(f"tokens: {[tok.decode([t]) for t in ids[0]]}")
    print(f"hidden_states entries: {len(out.hidden_states)} "
          f"(= n_layers {n_layers} + 1 embedding)")

    conv, d_raw, d_norm = detect_norm_convention(model, out.hidden_states,
                                                 out.logits)
    print(f"\nfinal-norm convention: {conv}")
    print(f"  raw diff    = {d_raw:.3e}")
    print(f"  normed diff = {d_norm:.3e}")
    print(f"  ratio       = {max(d_raw, d_norm) / max(min(d_raw, d_norm), 1e-30):.1e}"
          "   (<10 means the tie-break is arbitrary; treat as unknown)")
    if conv == "unknown":
        print("WARNING: neither reconstruction matched the real logits.")
        print("Likely causes: logit softcapping enabled, an untied lm_head,")
        print("or bf16 precision. Re-run with --dtype float32 first.")
        print("Do not trust the lens grid until this resolves.")

    if "lens" in want:
        section_lens(model, tok, out, ids, conv, max_pos=args.max_pos)
    if "delta" in want:
        section_delta(model, out, tok, ids, max_pos=args.max_pos)
    if "surprisal" in want:
        section_surprisal(out, tok, ids)
    if "attn" in want:
        section_attn(out, tok, ids, target=args.target, max_src=args.max_pos)
    if args.generate:
        section_generate(model, tok, ids, n_new=args.generate)


if __name__ == "__main__":
    main()
