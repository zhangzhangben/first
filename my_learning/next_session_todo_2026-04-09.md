# Next Session TODO (2026-04-09)

## Goal

Do not start a new model-improvement attempt immediately.
Continue stabilizing the unified stereo research framework first.

## First Read

Before doing anything else next time, re-open:

- `my_learning/unified_stereo_framework_summary_2026-04-09.md`
- `my_learning/baseline_contract_fastfoundation_booster_2026-04-09.md`
- `my_learning/repo_cleanup_inventory_2026-04-09.md`

## Current Core Files

The current framework center is:

- `core/stereo_datasets.py`
- `scripts/train_stereo_unified.py`
- `scripts/eval_stereo_unified.py`
- `scripts/visualize_stereo_results.py`

## Next Work Order

### 1. Evaluation layer first

Priority:

- make non-Booster evaluation rules clearer
- separate local heldout evaluation from official benchmark submission logic
- do not claim official benchmark comparability where it does not exist

Concrete next tasks:

- inspect KITTI official local-train evaluation vs submission workflow
- inspect Middlebury official local evaluation / submission workflow
- inspect ETH3D official local evaluation / submission workflow
- decide what should stay inside `eval_stereo_unified.py`
- decide what should be split into dataset-specific official submission scripts

### 2. Visualization layer second

Priority:

- make paper-friendly outputs easier to produce

Concrete next tasks:

- add standardized result panels
- add top-k worst sample export
- add optional no-occ visualization when masks exist
- design module-visualization hooks only after the evaluation layer is stable

### 3. Training protocol refinement third

Priority:

- make the training protocol more explicit and less ad hoc

Concrete next tasks:

- review which protocol items are baseline-fixed
- review which protocol items are dataset-specific
- review which protocol items are paper-specific
- avoid writing one new train script per paper idea again

### 4. Only then return to method improvement

Do not start a new module attempt until:

- baseline contract is stable
- unified train/eval flow is clear
- official-vs-local evaluation boundaries are written down

## Hard Rules For Next Time

- no new improvement patch without a stable protocol
- no training command without writing it down first
- no monitor metric treated as final conclusion
- no local heldout result described as official benchmark result
- no paper-inspired patch described as faithful reproduction unless it truly is

## One-Line Restart Plan

Next time, resume from the unified framework, finish evaluation clarity first, and only then revisit model improvement.
