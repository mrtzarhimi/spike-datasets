from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spike_datasets.binning import plan_view
from spike_datasets.evaluation import open_experiment
from spike_datasets.pairs import pair_experiments
from spike_datasets.splits import plan_splits
from spike_datasets.types import Source
from spike_datasets.xds import inspect_session, read_span as real_read_span
from tests.fixtures.xds import write_xds


def make_experiment(tmp_path, filename, *, dataset="chewie", mode="trial", strategy="chrono", seed=42, outer_fold=0):
    path = write_xds(tmp_path / filename, timestamps=np.arange(80) * .001,
                     include_force=dataset == "jango")
    source = Source(path, dataset, path.stem, "fixture")
    meta = inspect_session(source)
    from spike_datasets.types import TrialMeta
    meta = replace(meta, trials=tuple(TrialMeta(i, i * .008, (i + 1) * .008) for i in range(10)),
                   invalid_trial_ids=())
    view = plan_view(meta, mode=mode, bin_width_s=.004)
    manifest = plan_splits(view, strategy=strategy, seed=seed)
    return open_experiment(source, view, manifest, outer_fold)


def pair(tmp_path, **kwargs):
    source = make_experiment(tmp_path, "Chewie_20160928_01.mat", **kwargs)
    target = make_experiment(tmp_path, "Chewie_20160929_01.mat", **kwargs)
    return source, target, pair_experiments(source, target)


def test_zero_shot_does_not_read_target_before_score_and_methods_share_bins(tmp_path, monkeypatch):
    source, target, fixed = pair(tmp_path)
    reads = []
    def spy(read_source, span, *, with_behavior):
        reads.append((read_source.session_id, span, with_behavior))
        return real_read_span(read_source, span, with_behavior=with_behavior)
    monkeypatch.setattr("spike_datasets.evaluation.read_span", spy)

    source.inner_data(0, noise_seed=2)
    zero = fixed.lock_zero_shot({"noise_factor": 0.0, "method": "zero-shot"})
    assert not any(session == target.source.session_id for session, _, _ in reads)
    adaptation = fixed.adaptation_data(0, noise_seed=3)
    assert adaptation.fit.targets is None and adaptation.validation.targets is None
    assert not any(behavior for session, _, behavior in reads if session == target.source.session_id)
    adapted = fixed.lock_adaptation({"noise_factor": 0.0, "method": "adapted"})
    fixed.refit_adaptation(adapted, noise_seed=4)
    assert not any(behavior for session, _, behavior in reads if session == target.source.session_id)
    zero_score = fixed.score_zero_shot(zero, lambda data: data)
    adapted_score = fixed.score_adapted(adapted, lambda data: data)
    assert zero_score.value.group_ids == adapted_score.value.group_ids
    assert zero_score.value.spans == adapted_score.value.spans
    assert np.array_equal(zero_score.value.clean_counts, adapted_score.value.clean_counts)
    target_behavior_reads = [call for call in reads if call[0] == target.source.session_id and call[2]]
    assert target_behavior_reads
    assert target_behavior_reads


def test_adaptation_reads_only_target_held_in_spikes(tmp_path, monkeypatch):
    source, target, fixed = pair(tmp_path)
    reads = []
    def spy(read_source, span, *, with_behavior):
        reads.append((read_source.session_id, span, with_behavior))
        return real_read_span(read_source, span, with_behavior=with_behavior)
    monkeypatch.setattr("spike_datasets.evaluation.read_span", spy)
    data = fixed.adaptation_data(0, noise_seed=5)
    target_calls = [call for call in reads if call[0] == target.source.session_id]
    assert target_calls and not any(with_behavior for _, _, with_behavior in target_calls)
    held = set(target.outer_fold.held_out_group_ids)
    assert all(g not in held for g in data.fit.group_ids + data.validation.group_ids)
    assert not any(session == source.source.session_id and span.start in {
        s.start for g in source.view.groups if g.group_id in source.outer_fold.held_out_group_ids
        for block in g.blocks for s in block.spans
    } for session, span, _ in reads)


def test_pair_rejects_incompatible_recordings(tmp_path):
    source = make_experiment(tmp_path, "Chewie_20160928_01.mat")
    later = make_experiment(tmp_path, "Chewie_20160929_01.mat")
    reversed_day = make_experiment(tmp_path, "Chewie_20160927_01.mat")
    different_data = make_experiment(tmp_path, "Jango_20160929_01.mat", dataset="jango")
    different_view = make_experiment(tmp_path, "Chewie_20160929_01.mat", mode="fixed")
    different_strategy = make_experiment(tmp_path, "Chewie_20160929_01.mat", strategy="kfold")
    different_seed = make_experiment(tmp_path, "Chewie_20160929_01.mat", seed=7)
    with pytest.raises(ValueError, match="later"):
        pair_experiments(later, source)
    with pytest.raises(ValueError, match="dataset"):
        pair_experiments(source, different_data)
    with pytest.raises(ValueError, match="bin configuration"):
        pair_experiments(source, different_view)
    with pytest.raises(ValueError, match="split strategy"):
        pair_experiments(source, different_strategy)
    source_kfold = make_experiment(tmp_path, "Chewie_20160928_01.mat", strategy="kfold")
    with pytest.raises(ValueError, match="fold"):
        pair_experiments(source_kfold, make_experiment(
            tmp_path, "Chewie_20160929_01.mat", strategy="kfold", outer_fold=1
        ))
    with pytest.raises(ValueError, match="manifest"):
        pair_experiments(source, different_seed)
    with pytest.raises(ValueError, match="later"):
        pair_experiments(source, reversed_day)
