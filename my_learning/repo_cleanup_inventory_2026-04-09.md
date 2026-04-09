# Repo Cleanup Inventory (2026-04-09)

## Purpose

This file freezes the current repository state after the failed `UAG`, `LoS-lite`,
and `Mocha-MCCV` exploration lines.

The goal is to separate:

- mainline assets that should be kept
- failed exploration assets that should be frozen but not used as guidance
- legacy or temporary utilities that should not silently become the new default

This is a repository inventory, not a methods recommendation.

## Keep: Mainline Code Assets

These files are part of the current clean Fast-FoundationStereo baseline path and
should be treated as mainline code:

- `core/foundation_stereo.py`
- `core/update.py`
- `core/submodule.py`
- `core/extractor.py`
- `core/geometry.py`
- `Utils.py`
- `scripts/eval_booster.py`
- `scripts/submit_booster.py`
- `scripts/run_demo.py`
- `scripts/run_demo_tensorrt.py`
- `weights/23-36-37/model_best_bp2_serialize.pth`
- `weights/23-36-37/cfg.yaml`
- `configs/booster_scene_split_v1.yaml`
- `my_learning/project_memo_booster_fast.md`
- `AGENTS.md`

## Keep: Valid Baseline Experiment Assets

These are valid experiment/eval artifacts that are useful as references.

### Baseline local fine-tune assets

- `exp_booster_baseline_quick_v1/`
- `exp_booster_baseline_seq_quick_v1/`

Notes:

- `exp_booster_baseline_seq_quick_v1/` is the cleaner baseline local ablation asset.
- Its training record is useful, but its checkpoint format originally needed export
  before direct use by `eval_booster.py`.

### Baseline evaluation assets

- `output_booster_eval_paper_q416_v1/`
- `output_booster_eval_splitv1_baseline_q/`
- `output_booster_eval_splitv1_baseline_quick_v1/`
- `output_booster_eval_baseline_seq_quick_v1_valscenes/`

Notes:

- `output_booster_eval_paper_q416_v1/` is full-train analysis on all available train scenes.
- `output_booster_eval_splitv1_baseline_q/` is the split-heldout eval of the original pretrained baseline.
- `output_booster_eval_splitv1_baseline_quick_v1/` is the split-heldout eval of the older quick baseline line.
- `output_booster_eval_baseline_seq_quick_v1_valscenes/` is the heldout `val_scenes`
  eval of the sequence-loss baseline after exporting a serialized model.

These directories are not interchangeable. Their split scope and iteration settings differ.

## Freeze: Failed Exploration Assets

These assets should be frozen as failure records, not reused as the default next step.

The failed `exp_booster_loslite_seq_quick_v1/` experiment directory itself was deleted
during cleanup because its conclusions are already preserved in handoff records and it
should not remain in the active experiment workspace.

### Failed handoff / exploration documents

- `my_learning/handoffs/2026-04-07_booster_uag_handoff.md`
- `my_learning/handoffs/2026-04-09_loslite_sequence_handoff.md`

Why freeze:

- `UAG` and `LoS-lite` handoffs are failure records.
- These files are useful for postmortem reference, but they must not be treated as
  authoritative method guidance.

## Keep But Do Not Promote By Default

These files are utilities or legacy helpers. They can be useful, but they should not
silently define the new project workflow:

- no current keep-by-default temporary utility files are required after cleanup

Notes:

- Temporary smoke split files and one-off export helpers were deleted during cleanup.
- Serialized model export is now expected to be handled directly by the unified training script.

## Do Not Recreate

The following types of artifacts should not be recreated casually:

- new one-off training scripts for each paper/module
- experimental branches that replace the baseline main path without an identity-safe switch
- experiments that mix training-monitor metrics with final evaluation conclusions
- directories whose results are not tied to an explicit split/protocol description

## Current Working Rule

Before any new method work:

1. use the clean baseline code path
2. use the baseline contract document
3. make the method variable explicit
4. keep training-monitor metrics separate from independent eval results
