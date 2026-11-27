"""Run the nested within-session evaluation workflow on a local or synthetic XDS."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spike_datasets.binning import plan_view
from spike_datasets.evaluation import CleanData, open_experiment
from spike_datasets.splits import plan_splits
from spike_datasets.types import Source
from spike_datasets.xds import inspect_session

from _synthetic import write_recording


def _score(data: CleanData) -> float:
    valid = data.target_mask
    if data.targets is None or valid is None:
        raise ValueError("final scoring requires clean behavior targets")
    return float(np.square(data.targets[valid]).mean()) if valid.any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local XDS .mat file")
    parser.add_argument("--synthetic", action="store_true", help="create a temporary synthetic XDS")
    parser.add_argument("--view", choices=("trial", "fixed"), default="trial")
    parser.add_argument("--strategy", choices=("kfold", "chrono"), default="kfold")
    args = parser.parse_args()
    if bool(args.source) == args.synthetic:
        parser.error("choose exactly one of --source or --synthetic")
    context = TemporaryDirectory() if args.synthetic else nullcontext(None)
    with context as temp:
        path = write_recording(Path(temp) / "Chewie_20160928_01.mat", dataset="chewie") if args.synthetic else args.source
        source = Source(path, "chewie" if "Chewie" in path.name else "jango", path.stem, "synthetic" if args.synthetic else "local")
        view = plan_view(inspect_session(source), mode=args.view, bin_width_s=.02)
        manifest = plan_splits(view, strategy=args.strategy, seed=42)
        locked = []
        # Complete every fold's development and refit before revealing any outer score.
        for fold in manifest.outer_folds:
            experiment = open_experiment(source, view, manifest, fold.index)
            for inner in fold.inner_folds:
                experiment.inner_data(inner.index, noise_factor=.02, noise_seed=100 + fold.index * 10 + inner.index)
            selection = experiment.lock_selection({"noise_factor": .02, "model": "caller-defined", "model_seed": 7})
            refit = experiment.refit_data(selection, noise_seed=200 + fold.index)
            locked.append((experiment, selection, refit))
        for experiment, selection, refit in locked:
            score = experiment.score(selection, _score)
            print(f"fold={score.fold_index} bins={score.n_samples} refit_bins={refit.clean_counts.shape[0]} mse={score.value:.6g}")


if __name__ == "__main__":
    main()
