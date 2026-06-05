# AGENTS.md

## Project Rules

- Keep this file updated after each completed task.
- Use the project-local conda environment at `./.conda/epagcl-autoresearch`.
- First optimization target: Cora baseline with the fixed autoresearch command.
- Long-term research target: Defect-Aware Error-Passing Graph Contrastive Learning.
- Keep evaluation, split, and metric parsing fixed while comparing experiments.

## Environment

- Environment path: `./.conda/epagcl-autoresearch`
- Python: 3.10
- Expected CUDA stack for this machine:
  - `torch==2.5.1+cu124`
  - `torch_geometric`
  - `ogb`
  - `thop`

## Experiment Protocol

- Smoke command:

```bash
conda run -p ./.conda/epagcl-autoresearch python main.py --add_single --dataset Cora --epoch 10 --repeat 1 --eval 10 --eval_epoch 10 --seed 1 --emit_json
```

- Baseline command:

```bash
conda run -p ./.conda/epagcl-autoresearch python main.py --add_single --dataset Cora --epoch 500 --repeat 5 --eval 50 --eval_epoch 3000 --seed 1 --emit_json
```

- Main metric: `mean_acc` from the `AUTORESEARCH_RESULT` JSON line.
- Required diagnostics: `low_degree_acc`, `neighbor_inconsistency`, `false_negative_proxy`, augmentation edge statistics.
- Keep rule: keep a Cora experiment only when `mean_acc` improves the current best. If the gain is below 0.2, at least one diagnostic should also improve.
- Record all local experiments in `results.tsv`; do not commit `results.tsv`.

## Editable Boundaries

- During method search, editable files are `Model.py`, `Augmentation.py`, and the training logic in `main.py`.
- Do not change dataset loading, split semantics, evaluator metrics, or result JSON fields unless explicitly requested.
- Any aggressive model change must map to a paper claim or ablation, not just a leaderboard tweak.

## Research Directions

- Defect-aware adaptive augmentation: use embedding and degree signals to drop likely harmful edges and add likely useful candidate edges.
- Edge self-adversarial augmentation: generate budget-limited edge views that expose neighborhood inconsistency.
- False-negative-aware contrastive loss: downweight likely same-class or high-similarity negatives and add multi-positive terms.
- Stronger controlled encoder: residual GCN, GraphSAGE, GAT, LayerNorm projector, or gated fusion under fixed memory budget.

## Task Log

### 2026-06-05

- Created the EPAGCL autoresearch plan for a paper-oriented Defect-Aware EPAGCL direction.
- Initialized the experiment protocol around a fixed Cora baseline, machine-readable metrics, and mechanism diagnostics.

- Built project-local conda environment by cloning base after online `conda create` failed with `ProxyError`.
- Installed missing dependencies `ogb==1.3.6` and `thop==0.1.1.post2209072238` with pip legacy resolver.
- Fixed baseline harness issues: missing feature-drop flag, missing augmentation imports, optimizer signature mismatch, `train()` return mismatch, fixed split/seed controls, and machine-readable JSON output.
- Added diagnostics for low/mid/high degree accuracy, augmentation consistency, neighborhood inconsistency, false-negative proxy, and augmentation edge statistics.
- Smoke tests passed for Cora with `epoch=10` and `epoch=1`; both emitted `AUTORESEARCH_RESULT`.
- Formal fixed Cora baseline completed: mean_acc 84.4649, std_acc 1.1464, max_acc 86.2085, min_acc 82.7952.
