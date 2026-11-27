from collections import Counter
from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from spike_datasets.binning import plan_view
from spike_datasets.splits import load_manifest, plan_splits
from spike_datasets.types import RawSpan, SessionMeta, TrialMeta


def session(n_samples, bounds):
    return SessionMeta(
        dataset="chewie",
        session_id="2016-09-28",
        revision="fixture",
        native_bin_width_s=0.001,
        native_timestamps=np.arange(n_samples) * 0.001,
        unit_ids=("u1",),
        target_field="curs_v",
        target_units="cm/s",
        trials=tuple(TrialMeta(row, start * 0.001, stop * 0.001) for row, start, stop in bounds),
    )


def make_view(*weights, mode="trial"):
    bounds = []
    for index, weight in enumerate(weights):
        start = index * 8
        first_row = len(bounds)
        bounds.extend((first_row + row, start, start + 6) for row in range(weight))
    return plan_view(session(len(weights) * 8, bounds), mode=mode, bin_width_s=0.004)


def assert_roles(view, left, right, eligible):
    assert left and right
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(eligible)
    for role in (left, right):
        assert role == tuple(group_id for group_id in view.group_ids if group_id in role)

    raw_roles = [
        {
            row
            for group in view.groups if group.group_id in role
            for block in group.blocks
            for span in block.spans
            for row in range(span.start, span.stop)
        }
        for role in (left, right)
    ]
    assert all(raw_roles)
    assert raw_roles[0].isdisjoint(raw_roles[1])


def test_every_group_is_held_out_once_and_inner_is_nested():
    view = make_view(*([1] * 9))
    manifest = plan_splits(view)
    assert manifest.seed == 42
    assert manifest == plan_splits(view, seed=42)
    assert manifest.outer_folds != plan_splits(view, seed=17).outer_folds
    assert tuple(fold.index for fold in manifest.outer_folds) == (0, 1, 2)
    assert Counter(g for fold in manifest.outer_folds for g in fold.held_out_group_ids) == {
        group_id: 1 for group_id in view.group_ids
    }
    for fold in manifest.outer_folds:
        assert_roles(view, fold.fit_group_ids, fold.held_out_group_ids, view.group_ids)
        assert tuple(inner.index for inner in fold.inner_folds) == (0, 1)
        for inner in fold.inner_folds:
            assert_roles(view, inner.fit_group_ids, inner.validation_group_ids, fold.fit_group_ids)
        assert Counter(g for inner in fold.inner_folds for g in inner.validation_group_ids) == {
            group_id: 1 for group_id in fold.fit_group_ids
        }


@pytest.mark.parametrize(
    ("n_groups", "outer_fit", "held_out", "inner_fit", "validation"),
    [
        (3, (0, 1), (2,), (0,), (1,)),
        (4, (0, 1), (2, 3), (0,), (1,)),
        (6, (0, 1, 2, 3), (4, 5), (0, 1), (2, 3)),
        (7, (0, 1, 2, 3), (4, 5, 6), (0, 1), (2, 3)),
        (9, (0, 1, 2, 3, 4, 5), (6, 7, 8), (0, 1, 2, 3), (4, 5)),
    ],
)
def test_chronological_cutoffs_round_up_within_the_current_prefix(
    n_groups, outer_fit, held_out, inner_fit, validation
):
    view = make_view(*([1] * n_groups))
    (outer,) = plan_splits(view, strategy="chrono").outer_folds
    (inner,) = outer.inner_folds
    assert (outer.fit_group_ids, outer.held_out_group_ids) == (outer_fit, held_out)
    assert (inner.fit_group_ids, inner.validation_group_ids) == (inner_fit, validation)
    assert_roles(view, outer.fit_group_ids, outer.held_out_group_ids, view.group_ids)
    assert_roles(view, inner.fit_group_ids, inner.validation_group_ids, outer.fit_group_ids)


def test_chronological_roles_follow_view_order_for_nonconsecutive_group_ids():
    view = make_view(1, 1, 1)
    view = replace(view, groups=tuple(
        replace(group, group_id=group_id) for group, group_id in zip(view.groups, (70, 31, 54))
    ))
    (outer,) = plan_splits(view, strategy="chrono").outer_folds
    assert (outer.fit_group_ids, outer.held_out_group_ids) == ((70, 31), (54,))
    assert outer.inner_folds[0].validation_group_ids == (31,)


def test_group_packing_uses_eligible_trial_counts_in_both_views():
    weights = (10, 4, 3, 2, 1, 1)
    trial = plan_splits(make_view(*weights))
    fixed = plan_splits(make_view(*weights, mode="fixed"))
    assert trial.outer_folds == fixed.outer_folds
    assert trial.outer_folds[0].held_out_group_ids == (0,)
    held_out_weights = [sum(weights[g] for g in fold.held_out_group_ids) for fold in trial.outer_folds]
    assert sorted(held_out_weights) == [5, 6, 10]


def test_custom_kfold_count_preserves_coverage_and_nesting():
    view = make_view(1, 1, 1, 1)
    manifest = plan_splits(view, n_splits=4, seed=0)
    assert len(manifest.outer_folds) == 4
    assert {fold.held_out_group_ids for fold in manifest.outer_folds} == {(0,), (1,), (2,), (3,)}
    for fold in manifest.outer_folds:
        for inner in fold.inner_folds:
            assert_roles(view, inner.fit_group_ids, inner.validation_group_ids, fold.fit_group_ids)


@pytest.mark.parametrize(
    ("n_groups", "kwargs", "problem"),
    [
        (2, {}, "groups"),
        (2, {"strategy": "chrono"}, "groups"),
        (3, {"n_splits": 2}, "groups"),
        (3, {"n_splits": 4}, "groups"),
        (3, {"n_splits": 1}, "n_splits"),
        (3, {"n_splits": True}, "n_splits"),
        (3, {"seed": -1}, "seed"),
        (3, {"strategy": "shuffled"}, "strategy"),
    ],
)
def test_insufficient_groups_and_invalid_settings_fail(n_groups, kwargs, problem):
    with pytest.raises(ValueError, match=problem):
        plan_splits(make_view(*([1] * n_groups)), **kwargs)


def test_a_fixed_view_role_with_groups_but_no_bins_is_rejected():
    meta = session(24, [(0, 0, 4), (1, 5, 9), (2, 9, 15), (3, 16, 20)])
    view = plan_view(meta, mode="fixed", bin_width_s=0.004)
    assert tuple(group.n_bins for group in view.groups) == (1, 0, 1, 2)
    with pytest.raises(ValueError, match="no usable bins"):
        plan_splits(view, strategy="chrono")

    view = plan_view(meta, mode="trial", bin_width_s=0.004)
    plan_splits(view, strategy="chrono")



def test_replay_validates_bin_counts_but_preserves_groups_with_no_bins():
    meta = session(
        48, [(0, 0, 4), (1, 5, 9), (2, 9, 15), (3, 16, 20), (4, 24, 28), (5, 32, 36)]
    )
    view = plan_view(meta, mode="fixed", bin_width_s=0.004)
    manifest = plan_splits(view, strategy="chrono")
    assert view.groups[1].n_bins == 0
    assert manifest.outer_folds[0].inner_folds[0].fit_group_ids == (0, 1)
    assert load_manifest(manifest.to_json(), view) == manifest

    payload = json.loads(manifest.to_json())
    inner = payload["outer_folds"][0]["inner_folds"][0]
    inner["fit_group_ids"] = [1]
    inner["validation_group_ids"] = [0, 2, 3]
    with pytest.raises(ValueError, match="no usable bins"):
        load_manifest(json.dumps(payload), view)


def test_manifest_round_trip_is_canonical_and_persists_metadata_and_spans():
    view = plan_view(
        session(24, [(5, 0, 6), (6, 6, 7), (7, 8, 14), (9, 16, 22)]),
        mode="trial",
        bin_width_s=0.004,
    )
    manifest = plan_splits(view)
    text = manifest.to_json()
    payload = json.loads(text)
    assert payload["schema_version"] == 1
    assert payload["view"]["session"]["session_id"] == "2016-09-28"
    assert payload["view"]["session"]["revision"] == "fixture"
    assert payload["view"]["bin_width_s"] == 0.004
    assert payload["view"]["groups"][1]["owned_span"] == {"start": 8, "stop": 16}
    assert payload["view"]["excluded_trial_ids"] == [6]
    assert {"start": 4, "stop": 6} in payload["view"]["dropped_spans"]
    assert text == json.dumps(payload, sort_keys=True, separators=(",", ":"))
    reopened = load_manifest(json.dumps(payload, indent=2), view)
    assert reopened == manifest
    assert reopened.to_json() == text
    assert reopened.manifest_id == manifest.manifest_id
    assert plan_splits(view, seed=18).manifest_id != manifest.manifest_id


def test_longdouble_timestamp_padding_does_not_change_manifest_identity():
    dtype = np.dtype(np.longdouble)
    if dtype.itemsize <= 10 or np.finfo(dtype).nmant != 63 or sys.byteorder != "little":
        pytest.skip("this platform does not store padded x86 extended precision")
    n_samples = 60
    timestamps = np.arange(n_samples, dtype=dtype) * np.longdouble("0.001")
    raw = timestamps.view(np.uint8).reshape(n_samples, dtype.itemsize)
    first_bytes = raw.copy()
    second_bytes = raw.copy()
    first_bytes[:, 10:] = 0
    second_bytes[:, 10:] = 0xA5
    first_timestamps = first_bytes.view(dtype).reshape(-1)
    second_timestamps = second_bytes.view(dtype).reshape(-1)
    np.testing.assert_array_equal(first_timestamps, second_timestamps)

    trials = tuple(
        TrialMeta(row, start * 0.001, stop * 0.001)
        for row, start, stop in ((0, 0, 10), (1, 20, 30), (2, 40, 60))
    )
    meta = SessionMeta(
        dataset="chewie",
        session_id="longdouble-fixture",
        revision="fixture",
        native_bin_width_s=0.001,
        native_timestamps=first_timestamps,
        unit_ids=("u1",),
        target_field="curs_v",
        target_units="cm/s",
        trials=trials,
    )
    first_view = plan_view(meta, mode="trial", bin_width_s=0.004)
    second_view = plan_view(
        replace(meta, native_timestamps=second_timestamps), mode="trial", bin_width_s=0.004
    )

    first_manifest = plan_splits(first_view)
    second_manifest = plan_splits(second_view)

    assert second_manifest.to_json() == first_manifest.to_json()
    assert load_manifest(first_manifest.to_json(), second_view) == first_manifest


@pytest.mark.parametrize("changed", ["mode", "width", "session", "revision", "bins", "timestamps"])
def test_load_manifest_rejects_mismatched_view_or_session(changed):
    view = make_view(1, 1, 1)
    text = plan_splits(view).to_json()
    if changed == "mode":
        mismatched = plan_view(view.meta, mode="fixed", bin_width_s=0.004)
    elif changed == "width":
        mismatched = plan_view(view.meta, mode="trial", bin_width_s=0.002)
    elif changed == "session":
        mismatched = replace(view, meta=replace(view.meta, session_id="2016-09-29"))
    elif changed == "revision":
        mismatched = replace(view, meta=replace(view.meta, revision="changed"))
    elif changed == "timestamps":
        timestamps = view.meta.native_timestamps + 0.001
        mismatched = replace(view, meta=replace(view.meta, native_timestamps=timestamps))
    else:
        first, *rest = view.groups
        block = replace(first.blocks[0], spans=(RawSpan(1, 5),))
        mismatched = replace(view, groups=(replace(first, blocks=(block,)), *rest))

    with pytest.raises(ValueError, match="view.*identity"):
        load_manifest(text, mismatched)


@pytest.mark.parametrize(
    ("role_ids", "problem"),
    [([], "empty"), ([0, 0], "duplicate"), ([9], "unknown"), ([True], "integer"), ([1.0], "integer")],
)
def test_load_manifest_rejects_invalid_ids_in_a_role(role_ids, problem):
    view = make_view(1, 1, 1, 1)
    payload = json.loads(plan_splits(view).to_json())
    payload["outer_folds"][0]["held_out_group_ids"] = role_ids
    with pytest.raises(ValueError, match=problem):
        load_manifest(json.dumps(payload), view)


@pytest.mark.parametrize("changed", ["outer coverage", "inner membership", "inner coverage", "fold index"])
def test_load_manifest_rejects_invalid_partition_membership(changed):
    view = make_view(1, 1, 1)
    payload = json.loads(plan_splits(view).to_json())
    first, second, _ = payload["outer_folds"]
    if changed == "outer coverage":
        payload["outer_folds"][1] = {**first, "index": second["index"]}
    elif changed == "inner membership":
        first["inner_folds"][0]["fit_group_ids"] = first["held_out_group_ids"]
    elif changed == "inner coverage":
        first["inner_folds"][1] = {**first["inner_folds"][0], "index": 1}
    else:
        first["index"] = second["index"]

    with pytest.raises(ValueError, match="coverage|membership|index"):
        load_manifest(json.dumps(payload), view)


def test_load_manifest_rejects_chronological_roles_that_run_backwards():
    view = make_view(1, 1, 1)
    payload = json.loads(plan_splits(view, strategy="chrono").to_json())
    (outer,) = payload["outer_folds"]
    (inner,) = outer["inner_folds"]
    inner["fit_group_ids"], inner["validation_group_ids"] = (
        inner["validation_group_ids"], inner["fit_group_ids"]
    )
    with pytest.raises(ValueError, match="chronolog"):
        load_manifest(json.dumps(payload), view)


@pytest.mark.parametrize("overlap", ["ownership", "bins"])
def test_raw_span_overlap_is_rejected_when_planning_and_loading(overlap):
    view = make_view(1, 1, 1)
    text = plan_splits(view).to_json()
    first, second, third = view.groups
    if overlap == "ownership":
        first = replace(first, owned_span=RawSpan(0, 9))
    else:
        first = replace(first, blocks=(replace(first.blocks[0], spans=(RawSpan(8, 12),)),))
    overlapping = replace(view, groups=(first, second, third))

    with pytest.raises(ValueError, match="raw span"):
        plan_splits(overlapping)
    with pytest.raises(ValueError, match="raw span"):
        load_manifest(text, overlapping)


def test_chronological_planning_rejects_out_of_order_group_ownership():
    view = make_view(1, 1, 1)
    view = replace(view, groups=tuple(reversed(view.groups)))
    with pytest.raises(ValueError, match="chronolog"):
        plan_splits(view, strategy="chrono")


def test_load_manifest_rejects_invalid_json_and_schema_version():
    view = make_view(1, 1, 1)
    with pytest.raises(ValueError, match="JSON"):
        load_manifest("{", view)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest('{"schema_version": 1, "schema_version": 1}', view)
    payload = json.loads(plan_splits(view).to_json())
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(json.dumps(payload), view)
