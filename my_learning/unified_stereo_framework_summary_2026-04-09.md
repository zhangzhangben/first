# Unified Stereo Framework Summary (2026-04-09)

## Purpose

This note summarizes what was done to make the repository less chaotic and more
protocol-driven after the failed `UAG`, `LoS-lite`, and `Mocha-MCCV` attempts.

The goal of this round was **not** to improve the model again immediately.
The goal was to:

- restore a clean Fast-FoundationStereo mainline
- freeze failed exploration lines
- define a clearer baseline contract
- build a first-pass unified dataset / train / eval / visualization framework

This file is a progress summary, not a methods recommendation.

## What Was Restored / Removed

### Restored mainline direction

The repository was brought back to the clean Fast-FoundationStereo baseline path.
The failed `Mocha-MCCV` code path was removed from the active model path.

### Removed failed exploration artifacts

The following failed-line artifacts were removed from the active workspace:

- `core/mocha_mccv.py`
- `scripts/train_booster_mocha_mccv.py`
- `scripts/export_checkpoint_to_serialize.py`
- `scripts/train_booster_loslite.py`
- `scripts/train_booster_uag.py`
- `exp_booster_loslite_seq_quick_v1/`
- `exp_booster_mocha_mccv_only_strict_416q_v1/`
- `exp_booster_mocha_mccv_only_strict_smoke_416q_v1/`
- `exp_booster_mocha_mccv_seq_strict_416q_v1/`
- `exp_booster_mocha_mccv_seq_strict_smoke_416q_v1/`
- `output_booster_eval_mocha_mccv_only_strict_416q_v1_valscenes/`
- `configs/booster_scene_split_smoke_v1.yaml`
- `my_learning/mocha_fast_migration_design_2026-04-09.md`

### Kept baseline assets

These baseline experiment/eval directories were intentionally kept because they
are still useful reference assets:

- `exp_booster_baseline_quick_v1/`
- `exp_booster_baseline_seq_quick_v1/`
- `output_booster_eval_baseline_seq_quick_v1_valscenes/`
- `output_booster_eval_paper_q416_v1/`
- `output_booster_eval_splitv1_baseline_q/`
- `output_booster_eval_splitv1_baseline_quick_v1/`

They are now hidden from noisy `git status` output via `.gitignore`, but they
still exist on disk.

## Baseline Contract

The current baseline contract was written down in:

- `my_learning/baseline_contract_fastfoundation_booster_2026-04-09.md`

This fixes the current reference point for the local Booster-stage work:

- model family: `Fast-FoundationStereo`
- current baseline weight:
  - `weights/23-36-37/model_best_bp2_serialize.pth`
- current local development dataset:
  - `Booster`
- current local split:
  - `configs/booster_scene_split_v1.yaml`
- current primary local eval geometry:
  - `max_disp = 416`
  - `benchmark_gt_resolution = quarter`

Important distinction:

- training-loop monitor metrics are **not** final claims
- final local claims must use independent evaluation outputs

## New Framework Files Added

### Protocol files

- `configs/train_eval_protocol_fastfoundation_booster.yaml`
- `configs/train_eval_protocol_fastfoundation_sceneflow.yaml`
- `configs/train_eval_protocol_fastfoundation_kitti.yaml`
- `configs/train_eval_protocol_fastfoundation_middlebury.yaml`
- `configs/train_eval_protocol_fastfoundation_eth3d.yaml`

These are first-pass protocol templates, not final official benchmark recipes.

### Unified dataset layer

- `core/stereo_datasets.py`

This file now contains a common dataset registry for:

- `Booster`
- `SceneFlow`
- `KITTI`
- `Middlebury`
- `ETH3D`

### Unified training entry

- `scripts/train_stereo_unified.py`

This is the first-pass common training entry. It is intended to prevent the
repository from drifting into "one new train script per paper idea".

Current status:

- usable for `Booster`
- compatible with the new dataset registry
- non-Booster datasets are supported structurally, but their official evaluation
  logic is not yet fully integrated

### Unified evaluation entry

- `scripts/eval_stereo_unified.py`

Current behavior:

- `Booster` dispatches to the existing trusted `scripts/eval_booster.py`
- non-Booster datasets use a local heldout-style unified evaluator

Important limitation:

- for `SceneFlow / KITTI / Middlebury / ETH3D`, this is currently **local eval only**
- it is **not** equivalent to official submission / hidden-test evaluation
- each eval output directory should be treated together with its generated
  `evaluation_boundary.md`

### Unified visualization entry

- `scripts/visualize_stereo_results.py`

Current behavior:

- produces result panels with:
  - left image
  - GT disparity
  - predicted disparity
  - error map
- writes a `manifest.csv`

## Reference Repositories Used

This framework work was not written from scratch by intuition alone.
It was adapted with explicit reference to:

### RAFT-Stereo

Used mainly for:

- multi-dataset organization pattern
- training/evaluation structure
- local validation metric conventions

Reference files:

- `/tmp/RAFT-Stereo/core/stereo_datasets.py`
- `/tmp/RAFT-Stereo/train_stereo.py`
- `/tmp/RAFT-Stereo/evaluate_stereo.py`

### stereo_toolbox

Used mainly for:

- dataset-specific file layout details
- cleaner dataset-interface ideas
- disparity / error visualization helpers

Reference files:

- `/tmp/stereo_toolbox/stereo_toolbox/datasets_v2/sceneflow.py`
- `/tmp/stereo_toolbox/stereo_toolbox/datasets_v2/kitti.py`
- `/tmp/stereo_toolbox/stereo_toolbox/datasets_v2/middeval3.py`
- `/tmp/stereo_toolbox/stereo_toolbox/datasets_v2/eth3d.py`
- `/tmp/stereo_toolbox/stereo_toolbox/datasets/booster.py`
- `/tmp/stereo_toolbox/stereo_toolbox/visualization/disparity_map.py`
- `/tmp/stereo_toolbox/stereo_toolbox/visualization/error_map.py`

## Reference Mapping By File

This section makes the borrowing boundary more explicit.

- `core/stereo_datasets.py`
  - mainly follows the multi-dataset organization idea from `RAFT-Stereo`
  - dataset-specific path logic was adapted with help from `stereo_toolbox`
- `scripts/train_stereo_unified.py`
  - mainly follows the common train-loop shape from `RAFT-Stereo`
  - adapted to the current Fast-FoundationStereo model entry and Booster-stage protocol
- `scripts/eval_stereo_unified.py`
  - Booster delegation keeps the existing local formal eval path
  - non-Booster local metric shape was informed by `RAFT-Stereo/evaluate_stereo.py`
- `scripts/visualize_stereo_results.py`
  - result-panel direction was informed by `stereo_toolbox` visualization helpers

## Current Validation State

### What is already clear

- `LoS-lite` is a failed exploration line
- `UAG` is a failed exploration line
- `Mocha-MCCV only` is a failed exploration line
- the repository is now much closer to a "single baseline + single framework"
  state than before

### What is not yet complete

- official submission/eval workflows for `KITTI / Middlebury / ETH3D`
- a unified official multi-dataset evaluation layer
- module-level visualization hooks for research figures
- a final settled training protocol for future module-improvement experiments

## Current Risks / Limitations

1. The unified framework is still first-pass infrastructure, not a finalized
   benchmark system.

2. `scripts/train_stereo_unified.py` currently uses a common training shape, but
   that should not be confused with "official training protocol" for every dataset.

3. Non-Booster evaluation in `scripts/eval_stereo_unified.py` is local heldout
   evaluation only.

4. The repository still contains older modified tracked files unrelated to this
   framework round. They were not reverted automatically because they may reflect
   user-owned work.

## What Should Happen Next

The next sensible steps are:

1. keep improving the unified framework, not the model, until the workflow is stable
2. add clearer official-eval / submission handling for non-Booster datasets
3. add stronger visualization support for paper writing
4. only then return to new method exploration

## One-Sentence Summary

This round did **not** produce a new model improvement; it produced a cleaner,
more structured baseline-centered framework so future improvements can be tested
without repeating the previous chaos.
