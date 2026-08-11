# Method notes (all three experiments + vanilla)

## Shared recipe

See `configs/pretrain_d8_kaggle.json` and the README training table.  
Depth 8, 2×T4, fp16, 65,536 token batch, target 22k steps, ClimbMix data.

## E1 — L0 no attention, no residual (MLP ratio 4)

- `Block.skip_attn = (layer_idx == 0)`
- `Block.no_residual = (layer_idx == 0)`
- Forward L0: `x = MLP(norm(x))` with `ratio=4`
- Branch: `exp/layer0-no-attn-no-resid`

## E2 — same + MLP ratio 6

- Identical to E1 except L0 `MLP(..., ratio=6)`
- Intended to match full-block param count (`12 d²`) without attention
- Branch: `exp/layer0-no-attn-no-resid-mlp6`
- Run crashed near step 18400; use trajectory comparison, not only the last point

## E3 — L0 as E1 + middle-layer AttnBypassRouter

- L0 same as E1 (ratio 4)
- `middle_idx = n_layer // 2` (layer 4 on d8)
- `AttnBypassRouter`: `Linear(32 → 1)` on first 32 channels of residual stream → sigmoid gate
- Middle block: `x = x + gate * Attn(Norm(x))` then standard MLP residual
- Router params on AdamW (Muon rejects 1D / tiny matrices)
- Branch: `exp/layer0-no-attn-mid-router`

## Metrics

- **Train loss:** CE on training stream  
- **Val bpb:** bits/byte on held-out data (tokenizer-fair)

Lower is better for both.
