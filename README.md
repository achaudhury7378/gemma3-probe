# gemma3-probe

Structural and mechanistic-interpretability probes for Gemma 3.

Two scripts. The first tells you what the model *is*; the second tells you what
it *does* to a prompt.

---

## Status

Both scripts have been **exercised end to end against HF's real `Gemma3ForCausalLM`
class**, using a small randomly-initialised model (no checkpoint download). Every
section runs. Four bugs were found and fixed this way:

1. `out.attentions` is a **tuple of `None`s** under the sdpa backend, not `None`.
   The tuple is truthy, so the old guard passed and then raised `AttributeError`.
2. `from_pretrained(dtype=...)` only exists in transformers **v5**. v4 needs
   `torch_dtype=`. Both are now handled by version sniffing.
3. `hidden_states[0]` is the **unscaled** embedding, but Gemma multiplies by
   `sqrt(hidden_size)` before layer 0 runs. Uncorrected, the layer-0 residual
   delta reads ~8x too large (11.8 vs. the true 1.38).
4. The final-norm detector printed both candidate diffs at 4 decimal places, so a
   715,000x margin displayed as `0.0000` vs `0.0000` and looked like a coin flip.
   Now scientific notation plus an explicit ratio.

**They have still never seen real Gemma 3 weights.** Random weights validate the
plumbing, not the model. Every number these scripts emit about an actual
checkpoint is unverified until you run it.

Confirmed correct against the real config schema: module discovery locates
`model.layers`; layer classification reads `module.is_sliding` and returns the
5-local-then-1-global pattern starting local at index 0; `hidden_states` has
`n_layers + 1` entries; `hidden_states[-1]` is already post-final-norm.

One known version hazard: `config.sliding_window_pattern` is deprecated and
removed in transformers 4.55. Prefer `config.layer_types`, which the classifier
already tries first.

---

## `gemma3_probe.py` — static structure

Five sections, each runnable alone via `--sections`:

| section    | what it reports                                                    |
|------------|--------------------------------------------------------------------|
| `config`   | hyperparameters, and *which config attribute each one came from*    |
| `layers`   | per-layer local/global classification from three independent signals |
| `params`   | parameter accounting by component (embeddings / attn / MLP / norms) |
| `kv`       | KV-cache bytes, actual vs. hypothetical all-global                  |
| `hooks`    | residual / attention / MLP shapes and norms from a live forward pass |

```bash
# config-only, no weight download
python gemma3_probe.py --model google/gemma-3-1b-it \
    --sections config,kv --no-weights --seq-len 32768

# full inspection
python gemma3_probe.py --model google/gemma-3-1b-it --sections all --attn
```

The `kv` section is the most instructive and the cheapest to run. Try it at
`--seq-len 1024` and again at `--seq-len 131072` and watch the reduction figure
move. That delta is the entire argument for Gemma 3's attention interleave.

---

## `gemma3_trace.py` — forward-pass trace

### The framing that matters

Prefill is **parallel**, not sequential. Every prompt position passes through
every layer in a single forward pass; only the causal mask makes it "token by
token". So the object of interest is a **layers × positions grid**:

```
        positions →
  emb   [ what the embedding alone predicts at each position ]
  L00   [ ... ]
  ...
  L25   [ the actual prediction ]
```

Read a column downward to watch one position's prediction resolve. Read a row
across to see what a single layer contributes everywhere.

Genuinely sequential behaviour exists only during generation — that's `--generate`.

### Sections

| section     | what it shows                                                   |
|-------------|-----------------------------------------------------------------|
| `lens`      | logit lens grid: top prediction + probability per (layer, pos)   |
| `delta`     | relative residual change per layer per position                  |
| `surprisal` | per-position −log p of the actual next token, plus entropy       |
| `attn`      | attention into one chosen position, per layer                    |
| `--generate N` | the sequential loop, with top-3 alternatives per step         |

```bash
python gemma3_trace.py --model google/gemma-3-1b-it \
    --prompt "The capital of France is" \
    --sections lens,delta,surprisal --dtype float32
```

**Start in fp32.** In bf16 the numerical reconstruction check below will not
converge and you'll conclude the model is broken when it isn't.

---

## Design decisions

Two choices drive most of the code, and both exist for the same reason: HF's
Gemma 3 surface has moved across releases, and guessing at it produces silent
wrongness rather than errors.

**1. Discovery over hardcoding.** `find_decoder_layers()` walks `named_modules()`
looking for the first `ModuleList` whose children expose both `self_attn` and
`mlp`. Gemma 3 loads as flat `Gemma3ForCausalLM` for 1B / text-only checkpoints
and as nested `Gemma3ForConditionalGeneration` for 4B+ multimodal ones, so
`model.model.layers` breaks on roughly half the family. Config lookups use a
`probe()` helper that tries several candidate names and *prints which one
worked* — an `UNKNOWN` in the output is a finding, not a crash.

**2. Empirical calibration over asserted convention.** Logit lens breaks silently
if the final RMSNorm is applied wrongly, and whether `hidden_states[-1]` is
already normed varies by model and by transformers version.
`detect_norm_convention()` reconstructs the real logits both ways, compares
against `out.logits`, and uses whichever matches. If neither matches within
tolerance it refuses to vouch for the grid and names the likely causes.

---

## Gemma 3 architecture notes

Relevant facts, from the technical report (arXiv:2503.19786):

- Decoder-only transformer, GQA, RMSNorm applied both pre- and post-sublayer
- QK-norm replaces Gemma 2's attention soft-capping
- **5:1 interleaving of local/global attention layers**, starting with a local
  layer at index 0. Local layers use a 1,024-token sliding window
- RoPE base frequency: 10k on local layers, 1M on global layers
- 128K context (32K for the 1B model); 256k-entry vocabulary
- Sizes: 270M, 1B, 4B, 12B, 27B. 4B+ include a frozen SigLIP vision encoder;
  1B is text-only

The interleave exists to stop the KV cache exploding at long context. The `kv`
section quantifies that directly.

---

## Interpretive caveats

Worth internalising before drawing conclusions from any of this output:

- **Attention weight is not information flow.** Use attention maps to generate
  hypotheses, then confirm causally with activation patching. A head attending
  strongly to a token is not evidence that the token mattered.
- **Intermediate layers are off-distribution for the unembedding matrix**, which
  was only ever trained against the final layer's output. Early-layer logit lens
  output is often noise rather than a "belief the model holds". The tuned lens
  fixes this by fitting a per-layer affine probe; the raw lens here does not.
- **Averaging attention over heads hides specialisation.** Drop the `.mean(0)`
  once a layer looks interesting.
- **`--generate` recomputes the full prefix each step** with no KV cache. Correct
  for tracing, wrong for benchmarking.

---

## Going further

For semantic rather than structural analysis, Gemma Scope 2 provides sparse
autoencoders and transcoders trained on every layer of the Gemma 3 family
(270M through 27B, both pretrained and instruction-tuned), loadable via
`sae_lens`. There is also an interactive Neuronpedia demo for browsing and
steering features. These are released artifacts — no training required.

- https://ai.google.dev/gemma/docs/gemma_scope
- https://deepmind.google/models/gemma/gemma-scope/

---

## Requirements

```
torch
transformers
```

No plotting dependency. All output is text-mode so it works over SSH. Add
matplotlib yourself if you want heatmaps of the `lens` and `delta` grids —
they're the two that reward it.
