# LoS-lite Sequence-Loss Handoff

Date: 2026-04-09

Branch: `codex/booster-refine-prep`

## One-Paragraph Summary

Today we moved the LoS-lite work from an unstable / partially faithful prototype into a cleaner and more research-valid state. Two major issues were found and corrected:

1. the method logic was not yet faithful enough to the LoS paper's "reliable neighbors help uncertain pixels" idea, and
2. the training recipe was not aligned with the official refinement-style loss used by FoundationStereo / Fast-FoundationStereo.

After fixing both, we reran a fairer local comparison on the same Booster `split-v1` with the same `gamma=0.9` sequence-loss training criterion for both baseline and LoS-lite. Result: LoS-lite still does **not** beat the unified-loss baseline on `Bad2`, but the gap is now much smaller than before, and MAE is actually better.

## What Was Wrong Earlier

### 1. Earlier LoS-lite logic was not faithful enough

Earlier LoS-lite variants were not just "undertrained"; there were real method issues:

- propagation weighting did not cleanly prioritize lower-uncertainty neighbors
- LoS signals were previously mixed too directly into the motion/update path
- new structure logic could perturb baseline refinement too aggressively

This means older LoS-lite runs should **not** be treated as definitive evidence for or against the LoS idea.

### 2. Earlier training loss was not aligned with the paper's refinement supervision

`FoundationStereo` explicitly uses:

- `smooth L1` on initial disparity
- plus `gamma`-weighted iterative supervision on refined disparities

`Fast-FoundationStereo` explicitly uses, for refinement pruning retraining:

- `sum_k gamma^(K-k) * ||d_k - d||_1`
- plus feature distillation for the refinement retraining stage
- and it excludes the initial disparity supervision in that specific refinement retraining setup

Our earlier Booster LoS-lite training script supervised only the final prediction with a single masked `L1`, which was not close enough to the paper's refinement-style training logic.

## Code Changes Made Today

### A. LoS-lite logic was reworked to be more baseline-preserving and more faithful

Files:

- `core/update.py`
- `core/foundation_stereo.py`

Key changes:

- LoS-lite now behaves as "baseline update first, structure-guided propagation after", instead of directly modifying the motion encoder path.
- Reliable-neighbor weighting was changed to better reflect:
  - lower-uncertainty neighbors helping higher-uncertainty pixels
- propagation-sensitive math is now forced through a more stable fp32 path
- LoS-lite was given conservative initialization:
  - small residual influence
  - center-biased relation initialization
  - bounded residual correction
- a lightweight baseline-preserving regularizer is recorded during forward

### B. Training precision / stability was improved

Files:

- `Utils.py`
- `scripts/train_booster_loslite.py`

Key changes:

- AMP dtype is now configurable (`fp16` / `bf16` / `fp32`)
- current runs use `bf16`
- `GradScaler` is now only enabled when actual `fp16` is used

This fixed the earlier "forward finite but backward gradients explode / NaN" problem seen in full-model LoS-lite training.

### C. Training loss was corrected to use sequence supervision

File:

- `scripts/train_booster_loslite.py`

Key changes:

- added `--sequence_loss_gamma`
- added `masked_sequence_l1(preds, gt, valid, gamma)`
- training now uses:
  - sequence disparity loss over all iterative predictions
  - plus optional `loslite_preserve_weight * preserve_reg`
- logs now separate:
  - `train_loss`
  - `train_disp_loss`
  - `train_preserve_reg`

This is much closer to the official refinement-style supervision logic.

## Current Valid Comparison

The most important thing for the next Codex to know:

- old baseline quick finetune vs old LoS-lite quick finetune is **not** the fairest comparison anymore, because those runs used the old final-frame-only loss
- the new fair local comparison is:
  - baseline trained with the new sequence loss
  - LoS-lite trained with the same new sequence loss
  - same Booster split
  - same training budget
  - same formal validation metric

### Booster split used

File:

- `configs/booster_scene_split_v1.yaml`

Train scenes:

- 30 scenes

Val scenes:

- `Mirror`
- `Mouthwash`
- `Tablet`
- `TV`
- `BottledWater`
- `Oven2`
- `Motorcycle`
- `SoapDishes`

These are intentionally hard scenes.

## Latest Results That Still Matter

### 1. Zero-shot local baseline on split-v1 hard validation

Directory:

- `output_booster_eval_splitv1_baseline_q`

Important value:

- `overall/all Bad2 = 55.0878`

This is still useful as the zero-shot reference.

### 2. Baseline finetune with unified sequence loss

Directory:

- `exp_booster_baseline_seq_quick_v1`

History:

- epoch 0: `val_bad2_formal = 20.1088`, `val_mae_formal = 2.3171`
- epoch 1: `val_bad2_formal = 19.4001`, `val_mae_formal = 2.4652`
- epoch 2: `val_bad2_formal = 18.3045`, `val_mae_formal = 2.3202`

Current best local metric:

- `Bad2 = 18.3045`

### 3. LoS-lite finetune with unified sequence loss

Directory:

- `exp_booster_loslite_seq_quick_v1`

History:

- epoch 0: `val_bad2_formal = 19.6459`, `val_mae_formal = 2.3984`
- epoch 1: `val_bad2_formal = 19.1827`, `val_mae_formal = 2.1054`
- epoch 2: `val_bad2_formal = 18.8226`, `val_mae_formal = 2.0105`

Current best local metric:

- `Bad2 = 18.8226`

## How To Interpret The New Result

### What improved compared with earlier LoS-lite work

Compared with earlier LoS-lite variants:

- LoS-lite is now numerically stable
- LoS-lite is trained under a more paper-aligned sequence loss
- LoS-lite no longer lags baseline by a huge margin
- LoS-lite now achieves much better MAE than the sequence-loss baseline

### What is still not proven

LoS-lite still does **not** beat the sequence-loss baseline on `Bad2`:

- baseline sequence-loss best: `18.3045`
- LoS-lite sequence-loss best: `18.8226`

So the current honest conclusion is:

- LoS-lite is now much more defensible than before
- but it is still **not yet a winning method** on the primary local `Bad2` metric

### Interesting tension worth investigating

LoS-lite is worse on `Bad2` but better on `MAE`:

- baseline best MAE: `2.3202`
- LoS-lite best MAE: `2.0105`

This suggests the current LoS-lite may be reducing average error magnitude while still leaving too many pixels just above the 2-pixel threshold.

This is worth checking before discarding the method entirely.

## Which Older Results Are Now Obsolete As Main Evidence

These should be treated as historical/debug artifacts, not main research evidence:

- `exp_booster_loslite_smoke_v1`
- `exp_booster_loslite_quick_v1`
- `exp_booster_loslite_quick_v2`
- `exp_booster_loslite_quick_v2_lr3e4`
- `exp_booster_loslite_all_quick_v1`
- `exp_booster_loslite_all_quick_v2_bf16`
- `exp_booster_loslite_all_quick_v3_bf16`
- `exp_booster_loslite_losscheck_v1`
- `output_booster_eval_splitv1_loslite_quick_v1`
- `output_booster_eval_splitv1_loslite_quick_v2`
- `output_booster_eval_splitv1_loslite_quick_v2_lr3e4`

Reason:

- older method logic was not yet clean enough
- older training loss was not yet aligned enough

## Files Worth Keeping

These are the important code changes worth keeping and continuing from:

- `core/update.py`
- `core/foundation_stereo.py`
- `scripts/train_booster_loslite.py`
- `Utils.py`
- `.gitignore`

Potentially useful support files:

- `scripts/eval_booster.py`
- `scripts/submit_booster.py`
- `scripts/train_booster_uag.py`

Files not central to the current research result:

- `scripts/run_demo.py`
- `scripts/run_demo_tensorrt.py`
- `configs/booster_scene_split_smoke_v1.yaml`

## Engineering Notes

- experimental output directories are ignored by `.gitignore`
- do **not** `git add` `exp_*` or `output_*`
- `my_skill/` should also not be added unless explicitly desired

## Recommended Next Steps For The Next Codex

### 1. Run independent post-training evaluation for the new best checkpoints

Do not stop at training-history metrics. Convert or evaluate the new best checkpoints independently with the formal evaluation path.

Required targets:

- `exp_booster_baseline_seq_quick_v1/checkpoint_best.pth`
- `exp_booster_loslite_seq_quick_v1/checkpoint_best.pth`

Goal:

- produce new formal output directories comparable under the latest training recipe

### 2. Analyze why MAE improves but Bad2 still loses

This is the most promising scientific clue right now.

Check:

- per-scene delta
- balanced vs unbalanced breakdown
- whether LoS-lite reduces many large errors but fails to push enough pixels below the 2-pixel threshold

### 3. If continuing LoS-lite, prioritize threshold-aware refinement

Possible next directions:

- make propagation even more conservative on easy pixels
- adjust blend scheduling / alpha schedule
- explore loss terms that better reflect threshold metrics like Bad2

### 4. If pivoting away from LoS-lite, do it honestly

At this point, a pivot to another paper-backed direction such as GREAT-style context guidance would be scientifically defensible. But if that happens, keep this handoff and the sequence-loss baseline as the clean reference point.

## Bottom Line

Today ended in a much better place than it started:

- the old LoS-lite conclusions are no longer trusted
- the method logic is cleaner
- the training criterion is fairer
- the comparison is now substantially more valid

But the current best honest answer remains:

- LoS-lite is now close,
- it improves MAE,
- it is stable,
- but it still does **not** beat the unified-loss baseline on local `Bad2`.
