from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import h5py

from spike_datasets.binning import plan_view
from spike_datasets.evaluation import open_experiment
from spike_datasets.splits import plan_splits
from spike_datasets.types import Source, TrialMeta
from spike_datasets.xds import read_span as real_read_span
from tests.fixtures.xds import write_xds


def build(tmp_path, mode="trial", strategy="chrono", session_id="Chewie_20160928_01"):
    path = write_xds(tmp_path / f"{session_id}.mat", timestamps=np.arange(80) * 0.001)
    source = Source(path, "chewie", session_id, "fixture")
    # Ten distinct trial groups, with eight bins each on a 4 ms grid.
    from spike_datasets.xds import inspect_session
    meta = inspect_session(source)
    trials = tuple(TrialMeta(i, i * .008, (i + 1) * .008) for i in range(10))
    meta = replace(meta, trials=trials, invalid_trial_ids=())
    view = plan_view(meta, mode=mode, bin_width_s=.004)
    manifest = plan_splits(view, strategy=strategy)
    return source, view, manifest


def test_development_never_reads_outer_heldout_and_score_gets_only_clean_data(tmp_path, monkeypatch):
    source, view, manifest = build(tmp_path)
    calls = []
    def spy(read_source, span, *, with_behavior):
        calls.append((span, with_behavior))
        return real_read_span(read_source, span, with_behavior=with_behavior)
    monkeypatch.setattr("spike_datasets.evaluation.read_span", spy)

    experiment = open_experiment(source, view, manifest, 0)
    inner = experiment.inner_data(0, noise_seed=17)
    fold = manifest.outer_folds[0]
    held = {g.group_id for g in view.groups if g.group_id in fold.held_out_group_ids}
    assert not any(any(g.owned_span.start <= call[0].start < g.owned_span.stop for g in view.groups if g.group_id in held) for call in calls)
    assert set(inner.fit.group_ids) == set(fold.inner_folds[0].fit_group_ids)
    assert set(inner.validation.group_ids) == set(fold.inner_folds[0].validation_group_ids)
    selection = experiment.lock_selection({"noise_factor": .02, "alpha": 1})
    calls.clear()
    refit = experiment.refit_data(selection, noise_seed=23)
    assert set(refit.group_ids) == set(fold.fit_group_ids)
    before = len(calls)
    received = {}
    result = experiment.score(selection, lambda data: received.setdefault("data", data))
    assert len(calls) > before
    assert result.value is received["data"]
    assert all(g in held for g in result.value.group_ids)
    assert np.array_equal(result.value.clean_counts, np.rint(result.value.clean_counts))
    assert not np.array_equal(refit.features, refit.clean_counts.astype(float))


def test_selection_is_bound_to_source_manifest_and_fold(tmp_path):
    source, view, manifest = build(tmp_path, strategy="kfold")
    first = open_experiment(source, view, manifest, 0)
    selection = first.lock_selection({"noise_factor": 0.0})
    other = open_experiment(source, view, manifest, 1)
    with pytest.raises(ValueError, match="fold"):
        other.refit_data(selection, noise_seed=1)
    changed_source = replace(source, session_id="Chewie_20160929_01")
    with pytest.raises(ValueError, match="source"):
        open_experiment(changed_source, view, manifest, 0)
    changed_view = replace(view, meta=replace(view.meta, session_id=changed_source.session_id))
    changed_manifest = plan_splits(changed_view, strategy="kfold")
    changed_experiment = open_experiment(changed_source, changed_view, changed_manifest, 0)
    with pytest.raises(ValueError, match="source"):
        changed_experiment.refit_data(selection, noise_seed=1)
    other_manifest = plan_splits(view, strategy="chrono", seed=12)
    with pytest.raises(ValueError, match="manifest"):
        open_experiment(source, view, other_manifest, 0).refit_data(selection, noise_seed=1)


def test_validation_values_do_not_change_fit_noise(tmp_path):
    source, view, manifest = build(tmp_path)
    experiment = open_experiment(source, view, manifest, 0)
    original = experiment.inner_data(0, noise_seed=4)
    # Poison validation spikes and behavior on disk; fit-only statistics remain fixed.
    inner_fold = manifest.outer_folds[0].inner_folds[0]
    spans = [span for group in view.groups if group.group_id in inner_fold.validation_group_ids
             for block in group.blocks for span in block.spans]
    assert spans
    with h5py.File(source.path, "r+") as file:
        for span in spans:
            file["xds/spike_counts"][:, span.start:span.stop] = 1000
            file["xds/curs_v"][:, span.start:span.stop] = -900
    again = experiment.inner_data(0, noise_seed=4)
    np.testing.assert_array_equal(original.fit.noise_state.scales, again.fit.noise_state.scales)


def test_poisoning_outer_holdout_cannot_change_development_or_refit(tmp_path):
    source, view, manifest = build(tmp_path)
    experiment = open_experiment(source, view, manifest, 0)
    inner_before = experiment.inner_data(0, noise_seed=8)
    selection = experiment.lock_selection({"noise_factor": .02, "tag": "choice"})
    refit_before = experiment.refit_data(selection, noise_seed=19)
    membership = (manifest.outer_folds[0].fit_group_ids, manifest.outer_folds[0].held_out_group_ids)
    held_spans = [span for group in view.groups
                  if group.group_id in manifest.outer_folds[0].held_out_group_ids
                  for block in group.blocks for span in block.spans]
    with h5py.File(source.path, "r+") as file:
        for span in held_spans:
            file["xds/spike_counts"][:, span.start:span.stop] = 2000
            file["xds/curs_v"][:, span.start:span.stop] = -1900
    inner_after = experiment.inner_data(0, noise_seed=8)
    refit_after = experiment.refit_data(selection, noise_seed=19)
    assert membership == (manifest.outer_folds[0].fit_group_ids, manifest.outer_folds[0].held_out_group_ids)
    assert selection.settings == {"noise_factor": .02, "tag": "choice"}
    np.testing.assert_array_equal(inner_before.fit.noise_state.scales, inner_after.fit.noise_state.scales)
    np.testing.assert_array_equal(inner_before.validation.clean_counts, inner_after.validation.clean_counts)
    np.testing.assert_array_equal(refit_before.clean_counts, refit_after.clean_counts)
    np.testing.assert_array_equal(refit_before.features, refit_after.features)


@pytest.mark.parametrize("mode", ["trial", "fixed"])
@pytest.mark.parametrize("strategy", ["kfold", "chrono"])
def test_all_view_and_split_combinations_respect_role_reads(mode, strategy, tmp_path, monkeypatch):
    source, view, manifest = build(tmp_path, mode=mode, strategy=strategy)
    calls = []
    def spy(read_source, span, *, with_behavior):
        calls.append((span, with_behavior))
        return real_read_span(read_source, span, with_behavior=with_behavior)
    monkeypatch.setattr("spike_datasets.evaluation.read_span", spy)
    experiment = open_experiment(source, view, manifest, 0)
    assert experiment.outer_fold.index == 0
    experiment.inner_data(0, noise_seed=13)
    held_groups = [group for group in view.groups
                   if group.group_id in experiment.outer_fold.held_out_group_ids]
    def touches_held(span):
        return any(span.start < group.owned_span.stop and group.owned_span.start < span.stop
                   for group in held_groups)
    assert calls and not any(touches_held(span) for span, _ in calls)
    selection = experiment.lock_selection({"noise_factor": 0.0})
    experiment.refit_data(selection, noise_seed=14)
    assert not any(touches_held(span) for span, _ in calls)
    calls.clear()
    score = experiment.score(selection, lambda data: data)
    assert score.n_samples > 0
    assert calls and all(touches_held(span) for span, _ in calls)
