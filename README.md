# spike-datasets

`spike-datasets` provides shared, split-safe data access for Chewie and Jango XDS recordings. It plans trial or fixed-time bins, creates grouped outer and inner splits, and keeps learned preprocessing and noise calibration inside the corresponding fit partition. Model fitting and metric aggregation remain caller-owned.

## Installation

```bash
python -m pip install -e '.[test]'
```

## Within-session evaluation

Inspect one recording, make a view and split manifest, then open one experiment for each outer fold. Use `inner_data` to tune settings: its `fit` has clean counts plus one deterministic augmented `features` array, while `validation` remains clean. After selecting all settings, lock them, refit on the complete outer held-in partition, and score only through the final clean held-out interface.

```python
from spike_datasets.binning import plan_view
from spike_datasets.evaluation import open_experiment
from spike_datasets.splits import plan_splits
from spike_datasets.types import Source
from spike_datasets.xds import inspect_session

source = Source(path, "chewie", "Chewie_20160928_01", "local")
view = plan_view(inspect_session(source), mode="trial", bin_width_s=0.02)
manifest = plan_splits(view, strategy="kfold", n_splits=3, seed=42)
experiment = open_experiment(source, view, manifest, outer_fold=0)
inner = experiment.inner_data(0, noise_factor=0.02, noise_seed=101)
# Tune the caller's model with inner.fit.features and clean inner.validation.
selection = experiment.lock_selection({"noise_factor": 0.02, "model": "chosen-model"})
refit = experiment.refit_data(selection, noise_seed=201)
# Fit the frozen caller model on refit.features, then pass a scorer for clean held-out data.
score = experiment.score(selection, frozen_scorer)
```

`lock_selection` requires a finite `noise_factor` from 0.0 through 0.05. The clean integer `counts` are retained for count likelihoods; `features` are floating point and may include bounded noise. Validation and held-out counts and behavior remain clean. The seed is dedicated to augmentation and does not alter split membership.

## Fixed Day-0 → Day-K comparisons

Construct and open source and target experiments using the same bin configuration, split strategy, split seed, and fold index, then call `pair_experiments(day0, dayk)`. Day-K must be strictly later and in the same dataset. Zero-shot development uses Day-0 data only; `lock_zero_shot` creates a target-bound scoring token without reading target signals. Adaptation uses `adaptation_data`, which exposes target held-in spikes with behavior disabled, then `lock_adaptation` and `refit_adaptation`. Both score methods receive the same clean target held-out bins. Keep every fold and method choice locked before requesting any final score.

The package does not assign meaning to session-specific unit IDs or infer cross-day unit mappings. Callers must explicitly define that mapping and their model. Report paired method differences on the shared target bins and state the metric aggregation rule.

## Runnable examples

Run the examples against generated synthetic XDS recordings:

```bash
PYTHONPATH=src .venv/bin/python examples/within_session.py --synthetic
PYTHONPATH=src .venv/bin/python examples/cross_day.py --synthetic
```

Or pass local files with `--source PATH` for within-session work and `--day0 PATH --dayk PATH` for a pair. Both scripts accept `--view trial|fixed` and `--strategy kfold|chrono`. They use a small clean-target demonstration metric; replace that scorer and the caller-owned model fitting with the evaluation appropriate to your study. No real data is downloaded or included.
