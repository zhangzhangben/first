# Booster UAG Training Handoff

Date: 2026-04-07

## What Was Done

- Added a new Booster-specific training script at [scripts/train_booster_uag.py](/root/autodl-tmp/Fast-my/scripts/train_booster_uag.py).
- Added a local scene-level split config at [configs/booster_scene_split_v1.yaml](/root/autodl-tmp/Fast-my/configs/booster_scene_split_v1.yaml).
- Fixed the validation metric so training-time `val_bad2` and `val_mae` now follow the formal Booster evaluation logic from `eval_booster.py` instead of using a simplified train-resolution metric.
- Fixed supervision hygiene so `gt` and `valid` are no longer replicate-padded. Only input images are padded; predictions are cropped back before loss.
- Fixed unbalanced supervision alignment so if left/right sizes differ and left is resized to the right-view resolution, `gt` is resized and disparity values are scaled consistently.
- Fixed config recording so `train_args.yaml` is dumped after protocol defaults and runtime model args are resolved.
- Improved training history fields so the logged validation numbers are clearly marked as formal metrics.

## What Was Not Done

- No real training run was started.
- No smoke test epoch was run.
- No independent post-training check with `scripts/eval_booster.py` was run, because no new checkpoint exists yet.
- No export path was added from training checkpoints to a serialized model object for direct use by `eval_booster.py` or `submit_booster.py`.

## Current Intended Logic

- This script is for continued training from an existing serialized model passed by `--model_dir`.
- It freezes most of the model and trains only the uncertainty update gate.
- It is Booster-specific in validation semantics:
  - protocol defaults come from `eval_booster.py`
  - inference path for formal validation comes from `eval_booster.py`
  - quarter-resolution benchmark handling is Booster-specific
- The padding fix is generally correct beyond Booster, but the formal validation logic should not be assumed valid for other datasets without a dataset-specific evaluator.

## Important Caveats

- Training checkpoints saved by this script are checkpoint dicts, not necessarily the same format as the serialized model object expected by some inference scripts.
- The local split in `configs/booster_scene_split_v1.yaml` is for local ablation and validation only, not the official hidden Booster benchmark.
- Validation is now more rigorous but slower, because formal metric computation uses the formal inference path per sample.

## Recommended Next Steps

1. Run a small smoke test:
   - 1 epoch
   - small output directory
   - confirm `history.json`, `train_args.yaml`, `checkpoint_latest.pth`, and `checkpoint_best.pth` are produced
2. Inspect whether the saved checkpoint format needs an export/conversion step before running `eval_booster.py` or `submit_booster.py` directly.
3. Run a formal independent evaluation on the best checkpoint after training.
4. Only then start ablations on UAG hyperparameters.

## Suggested Smoke-Test Checklist

- Confirm the script starts from the intended base model.
- Confirm only uncertainty-gate parameters are trainable.
- Confirm `val_bad2_formal` and `val_mae_formal` appear in logs/history.
- Confirm no shape mismatch happens for unbalanced samples.
- Confirm the output directory contains:
  - `train_args.yaml`
  - `split_used.yaml`
  - `trainable_summary.json`
  - `history.json`
  - `checkpoint_latest.pth`
  - `checkpoint_best.pth` if validation improves

## Git Scope For This Work

Files intended for the commit:

- [scripts/train_booster_uag.py](/root/autodl-tmp/Fast-my/scripts/train_booster_uag.py)
- [configs/booster_scene_split_v1.yaml](/root/autodl-tmp/Fast-my/configs/booster_scene_split_v1.yaml)
- [my_learning/handoffs/2026-04-07_booster_uag_handoff.md](/root/autodl-tmp/Fast-my/my_learning/handoffs/2026-04-07_booster_uag_handoff.md)

Unrelated working-tree changes exist in other files and should be reviewed separately before committing them.
