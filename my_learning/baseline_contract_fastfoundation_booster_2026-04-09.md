# Baseline Contract: Fast-FoundationStereo on Booster (2026-04-09)

## Purpose

This document defines the current baseline contract for local Booster-stage research.

Anything claimed as an improvement must compare against this contract, or must clearly
state which parts differ.

This is a local research contract for the current stage of the project.
It is not the final multi-dataset contract for later `KITTI / Middlebury / ETH3D`.

## Baseline Model

- Model family: `Fast-FoundationStereo`
- Baseline serialized weight:
  - `weights/23-36-37/model_best_bp2_serialize.pth`
- Baseline config:
  - `weights/23-36-37/cfg.yaml`

## Current Local Dataset Stage

- Current train-time dataset: `Booster`
- Root path: `booster_gt`
- Current local split file:
  - `configs/booster_scene_split_v1.yaml`

Important:

- This split is for local ablation only.
- It must not be confused with the official Booster hidden test benchmark.
- Future stages may expand to `SceneFlow / KITTI / Middlebury / ETH3D`, but this contract
  only freezes the current Booster-stage local baseline.

## Split Definition

### Training scenes

- `Bathroom`
- `Bedroom`
- `Bottle`
- `Bottle1`
- `Bottles1`
- `Bucket`
- `Canteen`
- `Case`
- `CashBox`
- `CoffeeMaker`
- `Cooker1`
- `Cosmetics`
- `DogHouse`
- `Door`
- `ExtractorFan`
- `Fridge`
- `Lunch`
- `Microwave`
- `Moka`
- `Moka1`
- `OilCan`
- `Oven1`
- `Pots1`
- `Shower`
- `Sink`
- `TV1`
- `TV2`
- `Toilet`
- `Vodka`
- `Washer`

### Validation scenes

- `Mirror`
- `Mouthwash`
- `Tablet`
- `TV`
- `BottledWater`
- `Oven2`
- `Motorcycle`
- `SoapDishes`

## Baseline Inference / Eval Protocol

These are the fixed evaluation settings for current local comparisons:

- `paper_protocol: booster_q`
- `max_disp: 416`
- `benchmark_gt_resolution: quarter`
- `balanced_input_scale: 0.25`
- `unbalanced_input_scale: 1.0`
- `match_unbalanced_left_to_right: 1`
- `hiera: 0`

## Current Reference Evaluation Scripts

### Full-train analysis reference

- Script:
  - `scripts/eval_booster.py`
- Output:
  - `output_booster_eval_paper_q416_v1/`

Use:

- global problem diagnosis
- scene/class/setup inspection on all available train scenes

Do not use it as:

- the main independent heldout comparison for new methods

### Heldout validation reference

- Script:
  - `scripts/eval_booster.py`
- Output:
  - `output_booster_eval_baseline_seq_quick_v1_valscenes/`

Use:

- direct heldout comparison against future method experiments on the same `val_scenes`

Current key stats from this directory:

- `total_samples = 100`
- `overall/all Bad2 = 44.74095471992183`

Important:

- This heldout reference was evaluated with `valid_iters = 16`
- It is the most appropriate current comparison target for future local method experiments
  if those experiments also use `valid_iters = 16`

## Baseline Local Fine-Tune Reference

Current best clean local baseline experiment asset:

- `exp_booster_baseline_seq_quick_v1/`

Training record highlights:

- `epochs = 3`
- `batch_size = 1`
- `valid_iters = 8`
- `max_disp = 416`
- `quarter`
- `train_scope = all`
- `lr = 1e-4`
- `weight_decay = 0.0`

Best in-training heldout monitor result in this asset:

- `val_bad2_formal = 18.30447303704379`

Important:

- This number is from the training-loop monitor, not the same thing as the independent
  `eval_booster.py` result.
- Do not mix this monitor number with independent eval conclusions.

## What Counts As A Fair Improvement Claim

A new method may only be described as improved when all of the following are aligned:

- same baseline code path
- same split file
- same `max_disp`
- same `quarter/full` metric resolution
- same evaluation script
- same `valid_iters` at evaluation time
- same scene scope

If any of these differ, the result must be described as a different protocol, not a
direct improvement over the baseline contract.

## What Does Not Count As Final Evidence

The following do not count as final proof of improvement:

- training-loop `val_bad2_formal` by itself
- smoke runs
- experiments that only changed partial modules but changed the evaluation protocol
- full-train analysis results on all Booster train scenes

## Current Negative Controls / Failure Lines

These should not be treated as baseline variants:

- `exp_booster_loslite_seq_quick_v1/`
- any removed `Mocha-MCCV` experiment directories

Reason:

- they are failed or unconfirmed exploration lines
- they do not define the project baseline

## Working Rule For The Next Experiment

Before any new method implementation:

1. declare the exact training protocol
2. declare whether evaluation will use `valid_iters = 8` or `16`
3. declare the heldout comparison target directory
4. declare which result is only a monitor and which result is the final independent eval

