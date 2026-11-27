"""Run a fold-matched Day-0 → Day-K zero-shot/adaptation demonstration."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spike_datasets.binning import plan_view
from spike_datasets.evaluation import CleanData, open_experiment
from spike_datasets.pairs import pair_experiments
from spike_datasets.splits import plan_splits
from spike_datasets.types import Source
from spike_datasets.xds import inspect_session

from _synthetic import write_recording


def _score(data: CleanData) -> float:
    if data.targets is None or data.target_mask is None:
        raise ValueError("final scoring requires clean target behavior")
    return float(np.square(data.targets[data.target_mask]).mean()) if data.target_mask.any() else float("nan")


def _open(path: Path, dataset: str, mode: str, strategy: str):
    source = Source(path, dataset, path.stem, "local")
    view = plan_view(inspect_session(source), mode=mode, bin_width_s=.02)
    manifest = plan_splits(view, strategy=strategy, seed=42)
    return source, view, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day0", type=Path, help="local Day-0 XDS .mat file")
    parser.add_argument("--dayk", type=Path, help="local later-session XDS .mat file")
    parser.add_argument("--synthetic", action="store_true", help="create two temporary synthetic sessions")
    parser.add_argument("--view", choices=("trial", "fixed"), default="trial")
    parser.add_argument("--strategy", choices=("kfold", "chrono"), default="kfold")
    args = parser.parse_args()
    if args.synthetic:
        if args.day0 or args.dayk:
            parser.error("--synthetic cannot be combined with --day0/--dayk")
    elif not (args.day0 and args.dayk):
        parser.error("provide both --day0 and --dayk, or use --synthetic")
    context = TemporaryDirectory() if args.synthetic else nullcontext(None)
    with context as temp:
        if args.synthetic:
            day0_path = write_recording(Path(temp) / "Chewie_20160928_01.mat", dataset="chewie")
            dayk_path = write_recording(Path(temp) / "Chewie_20160929_01.mat", dataset="chewie")
        else:
            day0_path, dayk_path = args.day0, args.dayk
        dataset = "chewie" if "Chewie" in day0_path.name else "jango"
        source, source_view, source_manifest = _open(day0_path, dataset, args.view, args.strategy)
        target, target_view, target_manifest = _open(dayk_path, dataset, args.view, args.strategy)
        locked = []
        # Lock every fold and both methods before scoring any target held-out bins.
        for outer in range(len(source_manifest.outer_folds)):
            day0 = open_experiment(source, source_view, source_manifest, outer)
            dayk = open_experiment(target, target_view, target_manifest, outer)
            pair = pair_experiments(day0, dayk)
            for inner in day0.outer_fold.inner_folds:
                day0.inner_data(inner.index, noise_factor=.02, noise_seed=300 + outer * 10 + inner.index)
                pair.adaptation_data(inner.index, noise_factor=.02, noise_seed=400 + outer * 10 + inner.index)
            source_selection = day0.lock_selection({"noise_factor": .02, "model": "caller-defined", "seed": 9})
            zero_selection = pair.lock_zero_shot({"noise_factor": 0.0, "method": "zero-shot", "seed": 9})
            adapted_selection = pair.lock_adaptation({"noise_factor": .02, "method": "adapted", "seed": 9})
            source_refit = day0.refit_data(source_selection, noise_seed=500 + outer)
            target_refit = pair.refit_adaptation(adapted_selection, noise_seed=600 + outer)
            locked.append((pair, zero_selection, adapted_selection, source_refit, target_refit))
        for pair, zero_selection, adapted_selection, source_refit, target_refit in locked:
            zero = pair.score_zero_shot(zero_selection, _score)
            adapted = pair.score_adapted(adapted_selection, _score)
            assert zero.n_samples == adapted.n_samples
            print(f"fold={zero.fold_index} target_bins={zero.n_samples} source_refit={source_refit.clean_counts.shape[0]} adaptation_refit={target_refit.clean_counts.shape[0]} zero_shot_mse={zero.value:.6g} adapted_mse={adapted.value:.6g}")


if __name__ == "__main__":
    main()
