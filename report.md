# MP1 Report — Small Language Model Challenge

**Student ID:** 3036705993
**Repository:** https://github.com/kikihejiay-hub/dase7506mp1
**Final test BPB:** 1.5874

---

## 1. Introduction

This report describes a small GPT-style language model trained from random
initialization on WikiText-2 with the supplied BPE-2048 tokenizer. The goal
is to minimise full-test bits per byte (BPB) within a 5x baseline CPU
scoring budget, 4 GiB peak evaluation RAM, and 64 MiB of uncompressed
inference assets.

The classroom baseline is a 4-layer, 128-width, 4-head transformer with
learned absolute positions, Pre-LN, no dropout, and 1,088,256 parameters.
It reaches 2.1013 test BPB on this protocol.

Our final model reaches **1.5874 test BPB**, a 24.5% reduction, with a
27 MB checkpoint that evaluates in 24.4 s on CPU (FP32). Section 2 describes
the method. Section 3 describes the experimental setup. Section 4 reports
the baseline comparison and three ablations. Section 5 analyses why each
mechanism helps. Section 6 discusses cost and critical limitations.

---

## 2. Method

We keep the baseline's overall GPT structure (token embedding, causal
self-attention, GELU FFN, tied output head) and change three things.

### 2.1 Rotary position embedding (RoPE)

The baseline uses a learned absolute position table of shape
`[context, width]`. We replace it with RoPE, which rotates the query and
key vectors by position-dependent angles before the attention dot product.
RoPE is applied to Q and K only; values are unchanged. Because the
attention score then depends only on the relative offset between two
positions, the model no longer needs a separate position table, and the
parameter count drops by `context * width`.

We use the "split-halves" pairing: the first half of each head dimension
is rotated against the second half. Precomputed cosine and sine tables
cover the full context length.

### 2.2 Post-LN normalisation

The baseline uses Post-LN: `x = norm(x + sublayer(x))`, i.e. LayerNorm is
applied after each residual addition. We keep this arrangement rather than
switching to Pre-LN. Section 4 shows this choice is worth 0.026 BPB at
this model size.

### 2.3 Dropout 0.25

We add dropout with probability 0.25 to the attention output, the FFN
output, and the token embedding. The baseline has no dropout. Dropout is
active only during training; evaluation is unchanged.

### 2.4 Size and training

The final model is 8 layers, width 256, 8 heads, context 256, i.e.
6,834,688 parameters. Training uses AdamW with weight decay 0.1, a cosine
learning-rate schedule with 100-step linear warmup and peak lr 1e-3, batch
size 32, context 256, and 6,000 steps, giving 49,152,000 processed targets.
Gradient norm is clipped at 1.0.

---

## 3. Experimental setup

- **Data.** Supplied WikiText-2 raw splits and BPE-2048 tokenizer,
  unchanged. No external training text.
- **Training.** RTX 3080 Ti, bf16, ~280 s per run.
- **Evaluation.** CPU, FP32, supplied `evaluate.py`, protocol
  `7506-mp1-wt2-v2`. Validation split for all development and selection;
  test split only for the frozen final model.
- **Budget.** Baseline test scoring on this machine is 6.28 s, so the 5x
  ceiling is 31.4 s. Baseline validation scoring is 5.62 s, so the
  stricter ceiling is 28.1 s. Our final test evaluation is 24.4 s and
  satisfies both.
- **Search cost.** About ten training runs during development, roughly
  50 minutes of total GPU time.

---

## 4. Results

### 4.1 Baseline comparison

Both models are trained on the same 49,152,000 processed targets.

| Config | Params | Val BPB | Test BPB | Test eval |
|---|---|---|---|---|
| Baseline (128/4/4, Pre-LN, no dropout, abs pos) | 1,088,256 | 2.0711 | 2.1013 | 6.28 s |
| Final (256/8/8, Post-LN, RoPE, dropout 0.25) | 6,834,688 | 1.5682 | 1.5874 | 24.4 s |

The final model reduces test BPB by 0.514, at the cost of roughly 6.3x
parameters and 3.9x test scoring time. Both stay inside the resource
limits.

### 4.2 Ablations

Each ablation removes exactly one mechanism from the final model. All runs
use the same width, depth, seed, step count and processed targets.

| Config | Val BPB | Delta |
|---|---|---|
| Final (RoPE, Post-LN, dropout 0.25) | 1.5682 | — |
| A' — absolute position instead of RoPE | 1.6075 | +0.0393 |
| B' — dropout 0.0 | 1.5849 | +0.0167 |
| C' — Pre-LN instead of Post-LN | 1.5946 | +0.0264 |

All three mechanisms contribute. RoPE is the largest single contributor.

---

## 5. Analysis

### 5.1 Why RoPE helps

With a learned absolute table, the model must infer relative offsets from
absolute indices. With RoPE, the attention score is a function of
`(n - m)` by construction, so the inductive bias is built in. At this
scale, where the training set is small (49M targets) and the context is
256, that bias appears to be worth 0.039 BPB. Removing RoPE also adds
`256 * 256 = 65,536` parameters (the position table), so the comparison is
slightly unfavourable to RoPE in parameter count yet RoPE still wins.

### 5.2 Why dropout helps

Without dropout, training loss reaches 2.53 versus 2.92 with dropout, but
validation BPB is worse by 0.017. This is a clean overfitting signature:
the model fits the training windows more closely but generalises worse.
Dropout 0.25 reduces the gap. The effect is smaller than RoPE's but still
consistent across the model widths we tested.

### 5.3 Why Post-LN helps here

Pre-LN is the standard choice for deep transformers because it stabilises
training. At 8 layers and width 256, we do not see instability with
Post-LN, and Post-LN gives 0.026 BPB better validation. We attribute this
to Post-LN's stronger per-layer normalisation at shallow depth: each
sublayer output is renormalised before entering the next block, which
appears to help this small model more than it hurts optimisation. This is
a scale-dependent result and should not be generalised to larger models
without re-testing.

### 5.4 What the comparisons establish

The three ablations are independent: each removes one change while keeping
the other two. Their effects add roughly to the total gap versus baseline,
which suggests the mechanisms are not redundant. The paired baseline
comparison uses the same processed-target count, so the improvement is not
explained by seeing more data.

---

## 6. Cost, limitations, and critical analysis

### 6.1 Cost

| Item | Cost |
|---|---|
| Final training | ~280 s on RTX 3080 Ti, bf16 |
| Final test evaluation | 24.4 s CPU FP32 |
| Checkpoint | 27 MB uncompressed |
| Peak evaluation RAM | well below 4 GiB |
| Development search | ~10 runs, ~50 min GPU |

The model is smaller than the 64 MiB asset limit and evaluates in under
the 5x baseline ceiling under either baseline reference.

### 6.2 Limitations

- **Single seed.** All runs use seed 17. The 0.017–0.039 ablation deltas
  are larger than typical seed noise at this scale, but we did not measure
  seed variance, so the exact magnitudes should be treated as indicative.
- **Small search space.** We varied width and depth but did not sweep
  learning rate, dropout rate, or batch size. A larger sweep might find
  better settings, but would also increase search cost and risk
  overfitting validation.
- **Post-LN result is scale-specific.** The Post-LN advantage is observed
  at 8 layers and width 256. It may reverse at larger sizes.
- **No tokenizer or data changes.** We deliberately kept the supplied
  tokenizer and splits unchanged, as required.
- **One frozen test evaluation.** After freezing the final model, test was
  scored once.
  
### 6.3 Critical reflection

The largest gain comes from RoPE, which is a structural change, not from
scale. Scaling from 1.1M to 6.8M parameters contributes most of the raw
BPB reduction, but the three mechanism ablations show that architecture
choices also matter at fixed scale. The most surprising finding is that
Post-LN beats Pre-LN at this size, contrary to the usual guidance for
deep transformers. We report it as an empirical observation at this
scale, not as a general recommendation.

### 6.4 Reproducibility

The submitted checkpoint is self-contained. Running

    python evaluate.py --checkpoint runs/final-256/checkpoint.pt --device cpu --precision fp32 --split test

reproduces BPB 1.5874 in about 24 s on CPU. Training from scratch with the
command in `SUBMISSION.md` reproduces the checkpoint.

---

## 7. References

1. Su, J. et al. RoFormer: Enhanced Transformer with Rotary Position
   Embedding. arXiv:2104.09864, 2021.
2. Merity, S., Xiong, C., Bradbury, J., Socher, R. Pointer Sentinel
   Mixture Models. ICLR 2017. (WikiText-2)
3. Radford, A. et al. Language Models are Unsupervised Multitask Learners.
   OpenAI, 2019. (GPT-2 architecture conventions)
4. Loshchilov, I., Hutter, F. Decoupled Weight Decay Regularization.
   ICLR 2019. (AdamW)
5. Xiong, R. et al. On Layer Normalization in the Transformer
   Architecture. ICML 2020. (Pre-LN vs Post-LN)
