# MP1 Submission — Student ID 3036705993

DASE7506 Small Language Model Challenge.

## Final score

| Metric | Value |
|---|---|
| **Test BPB** | **1.5874** |
| Protocol | `7506-mp1-wt2-v2` |
| Test evaluation time (CPU, FP32) | 24.44 s |
| Checkpoint | `runs/final-256/checkpoint.pt` (27 MB) |
| Parameters | 6,834,688 |
| Baseline test BPB (reference) | 2.1013 |

## Method

A 256-width, 8-head, 8-layer GPT-style transformer trained from random
initialization. Three changes relative to the classroom baseline:

1. **RoPE** (rotary position embedding) replaces the learned absolute
   position embedding. RoPE is applied to Q and K only.
2. **Post-LN** normalization (LayerNorm after each residual addition),
   instead of the baseline's Pre-LN.
3. **Dropout 0.25** on the attention output, the FFN output, and the token
   embedding.

The FFN uses GELU, matching the baseline. Training uses AdamW with a
cosine schedule and 100-step warmup, 6000 steps of batch size 32 and
context 256, i.e. 49,152,000 processed training targets.

## Environment

- Python 3.12
- PyTorch 2.7.1+cu126 (training on GPU), 2.7.1+cpu (ranked evaluation)

## Install

    cd code
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v

For CPU-only evaluation, install torch==2.7.1 from the CPU index instead.

## Reproduce the final score

    python evaluate.py --checkpoint runs/final-256/checkpoint.pt --device cpu --precision fp32 --split test

Expected output: bpb = 1.5874, seconds ~24.

## Train from scratch

    python train.py --implementation student --config configs/medium.json --device cuda --precision bf16 --threads 4 --seed 17 --steps 6000 --run-dir runs/final-256

Training takes about 280 seconds on an RTX 3080 Ti (bf16).

## Experiments

### Baseline comparison (same number of processed targets)

| Config | Params | Val BPB | Test BPB | Test eval time |
|---|---|---|---|---|
| Baseline (128/4/4, Pre-LN, no dropout, absolute pos) | 1,088,256 | 2.0711 | 2.1013 | 6.28 s |
| Final (256/8/8, Post-LN, RoPE, dropout 0.25) | 6,834,688 | 1.5682 | 1.5874 | 24.44 s |

Both models were trained on 49,152,000 targets (6000 steps x 32 x 256).

### Ablations (all at 256 width, Post-LN, 6000 steps)

| Config | Val BPB | Delta vs final |
|---|---|---|
| Final model (RoPE, Post-LN, dropout 0.25) | 1.5682 | — |
| Ablation A' — absolute position instead of RoPE | 1.6075 | +0.0393 |
| Ablation B' — dropout 0.0 | 1.5849 | +0.0167 |
| Ablation C' — Pre-LN instead of Post-LN | 1.5946 | +0.0264 |

Interpretation:

- RoPE is the largest single contributor (+0.039 without it).
- Dropout 0.25 contributes +0.017; without it training loss is lower but validation BPB is worse (overfitting).
- Post-LN contributes +0.026 over Pre-LN.

### Evaluation-budget note

An earlier 288-width variant reached val BPB 1.542 but its test evaluation took 28.07 s, close to the 5x baseline ceiling. Based on evaluation-time margin (not test BPB), width was reduced to 256. Final test eval is 24.44 s, within 5x baseline under either reference.

## Cost disclosure

| Item | Cost |
|---|---|
| Final training | ~280 s on RTX 3080 Ti, bf16 |
| Final test evaluation | 24.44 s on CPU, FP32 |
| Checkpoint size | 27 MB (uncompressed) |

## AI assistance

The RoPE helper was written with AI assistance. The transformer, training recipe, ablations and analysis were developed by the author.

## Files

- student.py — final model
- student_postln_abs.py — ablation A'
- student_preln.py — ablation C'
- rope.py — RoPE helper
- model.py — classroom baseline
- train.py, evaluate.py, common.py — supplied scripts (unchanged)
- configs/medium.json — final config
- configs/nodrop256.json — ablation B' config
- runs/final-256/checkpoint.pt — final checkpoint
