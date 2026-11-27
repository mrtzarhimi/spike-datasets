"""Opt-in smoke checks against locally acquired Chewie and Jango recordings.

Set ``SPIKE_DATASETS_CHEWIE_ROOT`` and ``SPIKE_DATASETS_JANGO_ROOT`` to
directories containing the corresponding XDS ``.mat`` files to enable these
checks. Public CI intentionally leaves both unset.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

import pytest

from spike_datasets.binning import ViewPlan, plan_view
from spike_datasets.evaluation import CleanData, open_experiment
from spike_datasets.pairs import pair_experiments
from spike_datasets.splits import SplitManifest, plan_splits
from spike_datasets.types import RawSpan, Source
from spike_datasets.xds import discover_sessions, inspect_session


DATA_ROOTS = {
    "chewie": os.environ.get("SPIKE_DATASETS_CHEWIE_ROOT"),
    "jango": os.environ.get("SPIKE_DATASETS_JANGO_ROOT"),
}


@pytest.mark.parametrize("dataset", ("chewie", "jango"))
@pytest.mark.parametrize("mode", ("trial", "fixed"))
def test_real_dataset_release_smoke(dataset: str, mode: str) -> None:
    root_value = DATA_ROOTS[dataset]
    if not root_value:
        pytest.skip(f"set SPIKE_DATASETS_{dataset.upper()}_ROOT to enable real-data smoke test")
    root = Path(root_value).expanduser()
    if not root.is_dir():
        pytest.fail(f"configured {dataset} data root is not a directory: {root}")
    sources = discover_sessions(root, dataset, revision="local")
    if not sources:
        pytest.fail(f"no {dataset} XDS recordings found under {root}")

    summaries = []
    for source in sources:
        view, manifest = _plan(source, mode)
        replay = plan_splits(view, strategy="kfold", n_splits=3, seed=42)
        assert replay.to_json() == manifest.to_json()
        _assert_role_spans_are_disjoint(view, manifest)
        summaries.append(_summary(source, view, manifest))
    for summary in summaries:
        print(f"REAL_SMOKE {summary}")


def test_real_jango_fixed_day0_to_dayk_pair() -> None:
    root_value = DATA_ROOTS["jango"]
    if not root_value:
        pytest.skip("set SPIKE_DATASETS_JANGO_ROOT to enable real-data pair smoke test")
    sources = discover_sessions(Path(root_value).expanduser(), "jango", revision="local")
    candidates = []
    for source in sources:
        view, manifest = _plan(source, "fixed")
        if len(view.group_ids) >= 3:
            candidates.append((source, view, manifest))
    if len(candidates) < 2:
        pytest.fail("at least two Jango sessions with three fixed-view groups are required")

    day0_source, day0_view, day0_manifest = candidates[0]
    dayk_source, dayk_view, dayk_manifest = candidates[-1]
    day0 = open_experiment(day0_source, day0_view, day0_manifest, outer_fold=0)
    dayk = open_experiment(dayk_source, dayk_view, dayk_manifest, outer_fold=0)
    pair = pair_experiments(day0, dayk)
    selection = pair.lock_zero_shot({"noise_factor": 0.0, "method": "smoke"})

    def smoke_score(data: CleanData) -> float:
        assert data.targets is not None and data.target_mask is not None
        return float(data.targets[data.target_mask].mean())

    score = pair.score_zero_shot(selection, smoke_score)
    assert score.n_samples > 0
    print(
        "REAL_PAIR "
        f"dataset=jango view=fixed day0={day0_source.session_id} "
        f"dayk={dayk_source.session_id} target={dayk_view.meta.target_field} "
        f"day0_groups={len(day0_view.group_ids)} dayk_groups={len(dayk_view.group_ids)} "
        f"fold={score.fold_index} target_bins={score.n_samples}"
    )


def _plan(source: Source, mode: str) -> tuple[ViewPlan, SplitManifest]:
    view = plan_view(inspect_session(source), mode=mode, bin_width_s=0.02)
    manifest = plan_splits(view, strategy="kfold", n_splits=3, seed=42)
    return view, manifest


def _summary(source: Source, view: ViewPlan, manifest: SplitManifest) -> dict[str, object]:
    return {
        "dataset": source.dataset,
        "session_id": source.session_id,
        "view": view.mode,
        "native_bin_width_s": view.meta.native_bin_width_s,
        "target_field": view.meta.target_field,
        "eligible_groups": _compact_ids(view.group_ids),
        "eligible_group_count": len(view.group_ids),
        "fold_counts": [
            {
                "fold": fold.index,
                "fit_groups": len(fold.fit_group_ids),
                "held_out_groups": len(fold.held_out_group_ids),
                "inner": [
                    {
                        "fit_groups": len(inner.fit_group_ids),
                        "validation_groups": len(inner.validation_group_ids),
                    }
                    for inner in fold.inner_folds
                ],
            }
            for fold in manifest.outer_folds
        ],
    }


def _assert_role_spans_are_disjoint(view: ViewPlan, manifest: SplitManifest) -> None:
    for outer in manifest.outer_folds:
        _assert_no_shared_samples(
            _role_spans(view, outer.fit_group_ids),
            _role_spans(view, outer.held_out_group_ids),
        )
        for inner in outer.inner_folds:
            _assert_no_shared_samples(
                _role_spans(view, inner.fit_group_ids),
                _role_spans(view, inner.validation_group_ids),
            )


def _role_spans(view: ViewPlan, group_ids: Iterable[int]) -> tuple[RawSpan, ...]:
    by_id = {group.group_id: group for group in view.groups}
    return tuple(
        span
        for group_id in group_ids
        for block in by_id[group_id].blocks
        for span in block.spans
    )


def _assert_no_shared_samples(left: tuple[RawSpan, ...], right: tuple[RawSpan, ...]) -> None:
    ordered_left = sorted(left, key=lambda span: span.start)
    ordered_right = sorted(right, key=lambda span: span.start)
    i = j = 0
    while i < len(ordered_left) and j < len(ordered_right):
        a, b = ordered_left[i], ordered_right[j]
        assert a.stop <= b.start or b.stop <= a.start, f"raw-sample overlap: {a} and {b}"
        if a.stop <= b.start:
            i += 1
        else:
            j += 1


def _compact_ids(values: tuple[int, ...]) -> str:
    if not values:
        return ""
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append(f"{start}-{previous}" if start != previous else str(start))
            start = value
        previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)
