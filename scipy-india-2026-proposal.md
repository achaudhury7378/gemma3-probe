# SciPy India 2026 — Talk Proposal (draft)

**Session type:** Talk (30 minutes, conference day, 20 December 2026)
**Suggested track:** Reproducibility in research
*(secondary fit: AI, machine learning, and data-driven discovery)*
**Submit at:** https://cfp.scipy.in/scipy-india-2026/
**Deadline:** 19 October 2026, 23:59 IST — reviewed on a rolling basis, so submit early

> Note: field names below follow the usual pretalx layout. Check the actual form;
> it may ask for different or additional fields.

---

## Title

Pick one:

1. **Your Model Instrumentation Is Probably Wrong: Building Self-Verifying Probes for Transformers**
2. **Looking Inside a Language Model with Plain PyTorch — and the Four Ways It Silently Lies to You**
3. **Reading a Transformer's Internals on a Laptop: Instrumentation, and How to Know It Worked**

Option 1 leads with the problem and is the most likely to get read past the title.
Option 3 is the safest if the reviewers skew towards teaching and outreach.

---

## Abstract

*(~100 words — this appears in the schedule)*

Inspecting what a language model does internally needs no specialised framework:
`output_hidden_states`, `output_attentions`, and PyTorch forward hooks are enough.
What it does need is a way to tell whether your instrumentation is correct — because
when it is wrong, it usually does not crash. It returns plausible numbers.

This talk walks through building a structural and behavioural probe for Google's
Gemma 3 in plain PyTorch, and through four real bugs found while testing it: one
that crashed, and three that quietly produced believable, wrong output. The fix in
each case was to make the probe verify its own assumptions at runtime rather than
assert them.

---

## Description

*(~450 words — reviewers rank on this)*

**Motivation.** A growing number of scientific Python users now run transformer
models as part of their pipeline, but comparatively few inspect what happens inside
one. The tooling reputation is that this requires a dedicated framework. It does
not. Hugging Face `transformers` exposes the residual stream via
`output_hidden_states`, the attention matrices via `output_attentions`, and every
submodule through standard PyTorch forward hooks. Three primitives, and you can
read a model's intermediate state directly.

The hard part is not access. It is correctness.

**The problem this talk is about.** Model instrumentation has an unusually bad
failure mode: wrong setups mostly do not raise. They return arrays of the right
shape, full of numbers that look reasonable, and you draw a conclusion from them.
Three of the four bugs described here behaved exactly that way.

**Case study.** I will build up a probe for Gemma 3, chosen because its architecture
is legible in the output. Gemma 3 interleaves five local sliding-window attention
layers with one global layer, so the attention maps show a band-diagonal stripe in
five of every six layers and a full causal triangle in the sixth. The KV-cache
arithmetic explains why the design exists, and takes about ten lines to compute.

Then a layers-by-positions trace of a single forward pass — the logit lens grid —
with the framing that matters: prefill is parallel, not sequential. Every prompt
position passes through every layer in one pass; only the causal mask makes it look
sequential. So "watching a model think" is reading a grid, not animating a loop.

**The four failures.** Each gets a slide, and each generalises well beyond Gemma:

- `output_attentions` returns a *tuple of `None`s* under the SDPA backend rather
  than `None`. The tuple is truthy, so the obvious guard passes.
- Hugging Face renamed `from_pretrained(torch_dtype=)` to `(dtype=)` at v5. Code
  written against one major version fails on the other.
- `hidden_states[0]` is the *unscaled* embedding, but Gemma multiplies by
  `sqrt(hidden_size)` before layer 0 runs. Uncorrected, the layer-0 residual delta
  reads roughly eight times too large — and reads as a finding about the model.
- Whether `hidden_states[-1]` has already had the final norm applied varies by
  model and library version. Get it wrong and the logit lens produces a coherent
  but meaningless grid.

**The resolution.** Rather than asserting these conventions, the probe detects them.
It reconstructs the model's real logits both ways and uses whichever matches; it
locates the decoder by walking the module tree instead of hardcoding a path; it
reports *which* config attribute each value came from. When it cannot verify an
assumption, it says so instead of returning a number.

**Takeaway.** A reusable pattern for instrumenting code you did not write, and a
laptop-runnable repository. Everything runs on CPU with a 1B model. No GPU required.

---

## Outline (30 minutes)

| Time | Section |
|------|---------|
| 0–4   | Why inspect internals at all; the three primitives |
| 4–10  | Gemma 3's local/global interleave, made visible; KV-cache arithmetic |
| 10–17 | The layers × positions grid; prefill is parallel, not sequential |
| 17–25 | Four ways this silently returns wrong answers |
| 25–28 | Self-verifying probes as a general pattern |
| 28–30 | Where to go next (Gemma Scope 2, tuned lens); questions |

---

## Audience and prerequisites

Working Python and basic NumPy. Familiarity with PyTorch helps but is not assumed —
hooks are explained from scratch. No prior transformer internals knowledge needed;
the architecture is introduced through the plots rather than through equations.

Useful to: anyone instrumenting a library they did not write, anyone using
transformer models in a research pipeline, and anyone who has ever trusted an
array because it had the right shape.

---

## Links

- Repository: https://github.com/achaudhury7378/gemma3-probe
- Gemma 3 technical report: arXiv:2503.19786

---

## Speaker bio

*(verify and edit — written from what I know, and it may be out of date)*

Abhi is a data engineer and consultant at PwC in Pune, working on optimisation
modelling, data engineering, and analytics pipelines. He holds an MSc in Operations
Research from IIT Bombay and has two filed patents from earlier work on MILP-based
optimisation pipelines at TCS. He has contributed to the open-source solver
ecosystem and builds interpretability and benchmarking tooling for local language
models outside of work.

---

## Before you submit — open items

1. **Run the scripts against real Gemma 3 weights.** Right now they have only been
   exercised against a randomly-initialised model. Everything above is defensible,
   but a reviewer who checks the repo will see the caveat, and you cannot give the
   talk without real plots. This is the highest-priority item.
2. **Employer approval.** Talks given with a PwC affiliation may need internal
   sign-off. Worth checking before the abstract is public.
3. **Speaking evidence.** First-time speakers are explicitly welcomed, so this is
   not disqualifying, but the repository plus a short screen recording of the tool
   running would strengthen a thin speaking history.
4. **Track choice.** I have suggested Reproducibility over AI/ML deliberately —
   the AI track will be the most crowded, and the correctness angle is the
   differentiated one. Reconsider if the CFP form indicates otherwise.
