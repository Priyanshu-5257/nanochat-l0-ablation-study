# Method notes

## Ablation definition

Only **layer index 0** is modified:

1. **No multi-head attention** — `Block.attn is None`; forward never calls attention.
2. **No residual connections** — block output is `MLP(Norm(x))`, not `x + MLP(Norm(x))`.

All other layers keep the standard Pre-LN residual transformer block used by nanochat.

## Why ablate both together?

This study answers: *“Does the standard L0 block (attn+res+mlp+res) matter vs a pure local MLP rewrite?”*

It does **not** disentangle attention vs residual. A follow-up factorial design would be:

| ID | L0 Attn | L0 residual |
|----|---------|-------------|
| A | yes | yes (vanilla) |
| B | no | yes |
| C | yes | no |
| D | no | no (this study) |

## Metrics

- **Train loss**: mean cross-entropy on the training stream (logged each step).
- **Val bpb**: bits per byte on held-out data — tokenizer-fair compression metric used by nanochat (`evaluate_bpb`).

Lower is better for both.

## Hardware caveats

- T4 has no bf16; training used **fp16 + GradScaler**.
- FlashAttention-3 is unavailable on T4 → SDPA fallback (slower, but same math intent).
- Free Kaggle sessions limit wall-clock; 22k steps fit ~6h.
