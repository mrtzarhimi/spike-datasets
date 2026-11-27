"""Plan and replay nested, group-isolated splits from view metadata."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from struct import Struct
from typing import Literal

import numpy as np

from .binning import ViewGroup, ViewPlan

SplitStrategy = Literal["kfold", "chrono"]
_SCHEMA_VERSION = 1
_RAW_BOUNDS = Struct("<QQ")


@dataclass(frozen=True)
class InnerFold:
    """One development split within an outer fold's held-in groups."""

    index: int
    fit_group_ids: tuple[int, ...]
    validation_group_ids: tuple[int, ...]


@dataclass(frozen=True)
class OuterFold:
    """One final holdout and its nested development splits."""

    index: int
    fit_group_ids: tuple[int, ...]
    held_out_group_ids: tuple[int, ...]
    inner_folds: tuple[InnerFold, ...]


@dataclass(frozen=True)
class SplitManifest:
    """An immutable set of splits bound to one session and binning plan."""

    strategy: SplitStrategy
    n_splits: int
    seed: int
    outer_folds: tuple[OuterFold, ...]
    _view_json: str

    @property
    def manifest_id(self) -> str:
        """SHA-256 of the canonical UTF-8 manifest."""
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialize the versioned manifest as compact JSON with sorted keys."""
        return _canonical_json({
            "schema_version": _SCHEMA_VERSION,
            "strategy": self.strategy,
            "n_splits": self.n_splits,
            "seed": self.seed,
            "view": json.loads(self._view_json),
            "outer_folds": [asdict(fold) for fold in self.outer_folds],
        })


def plan_splits(
    view: ViewPlan, *, strategy: SplitStrategy = "kfold", n_splits: int = 3, seed: int = 42
) -> SplitManifest:
    """Plan grouped folds or one forward split, with nested development roles.

    ``n_splits`` and ``seed`` control grouped K-fold. Chronological boundaries
    always reserve the latest third, rounded up, at both levels.
    """
    _validate_settings(strategy, n_splits, seed)
    view_json = _view_identity(view)
    if strategy == "chrono":
        folds = (_chrono_fold(view.group_ids),)
    else:
        rng = np.random.default_rng(seed)
        by_id = {group.group_id: group for group in view.groups}
        outer_folds = []
        for index, (fit, held_out) in enumerate(_pack_groups(view.groups, n_splits, rng)):
            held_in = tuple(by_id[group_id] for group_id in fit)
            inner_folds = tuple(
                InnerFold(inner_index, inner_fit, validation)
                for inner_index, (inner_fit, validation) in enumerate(_pack_groups(held_in, 2, rng))
            )
            outer_folds.append(OuterFold(index, fit, held_out, inner_folds))
        folds = tuple(outer_folds)
    manifest = SplitManifest(strategy, n_splits, seed, folds, view_json)
    _validate_folds(manifest, view)
    return manifest


def load_manifest(text: str, view: ViewPlan) -> SplitManifest:
    """Load persisted assignments and validate them against the supplied view."""
    try:
        payload = json.loads(text, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as error:
        raise ValueError("invalid split manifest JSON") from error
    payload = _record(
        payload, {"schema_version", "strategy", "n_splits", "seed", "view", "outer_folds"},
        "split manifest",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported split manifest schema_version: {version!r}")
    strategy, n_splits, seed = payload["strategy"], payload["n_splits"], payload["seed"]
    _validate_settings(strategy, n_splits, seed)
    view_json = _view_identity(view)
    if _canonical_json(payload["view"]) != view_json:
        raise ValueError("manifest view and session identity does not match the supplied view")
    folds = tuple(_read_outer_fold(value) for value in _array(payload["outer_folds"], "outer folds"))
    manifest = SplitManifest(strategy, n_splits, seed, folds, view_json)
    _validate_folds(manifest, view)
    return manifest


def _validate_settings(strategy: SplitStrategy, n_splits: int, seed: int) -> None:
    if strategy not in ("kfold", "chrono"):
        raise ValueError(f"unsupported split strategy: {strategy!r}")
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _pack_groups(
    groups: tuple[ViewGroup, ...], count: int, rng: np.random.Generator
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if len(groups) < count:
        raise ValueError(f"at least {count} groups are needed for {count} grouped folds")
    by_weight = [groups[index] for index in rng.permutation(len(groups))]
    by_weight.sort(key=lambda group: len(group.eligible_trial_ids), reverse=True)
    holdouts: list[set[int]] = [set() for _ in range(count)]
    weights = [0] * count
    for index, group in enumerate(by_weight):
        # Seed each fold with one group before adding to the lightest fold.
        fold = index if index < count else min(range(count), key=weights.__getitem__)
        holdouts[fold].add(group.group_id)
        weights[fold] += len(group.eligible_trial_ids)
    group_ids = tuple(group.group_id for group in groups)
    return [
        (tuple(g for g in group_ids if g not in held_out), tuple(g for g in group_ids if g in held_out))
        for held_out in holdouts
    ]


def _chrono_fold(groups: tuple[int, ...]) -> OuterFold:
    if len(groups) < 3:
        raise ValueError("chronological splitting needs at least three groups")
    outer_cutoff = len(groups) - (len(groups) + 2) // 3
    fit = groups[:outer_cutoff]
    inner_cutoff = len(fit) - (len(fit) + 2) // 3
    inner = InnerFold(0, fit[:inner_cutoff], fit[inner_cutoff:])
    return OuterFold(0, fit, groups[outer_cutoff:], (inner,))


def _validate_folds(manifest: SplitManifest, view: ViewPlan) -> None:
    group_ids = view.group_ids
    positions = {group_id: index for index, group_id in enumerate(group_ids)}
    bin_counts = {group.group_id: group.n_bins for group in view.groups}
    chrono = manifest.strategy == "chrono"
    outer_count = 1 if chrono else manifest.n_splits
    inner_count = 1 if chrono else 2
    if len(manifest.outer_folds) != outer_count:
        raise ValueError(f"manifest must contain {outer_count} outer folds")

    outer_coverage: Counter[int] = Counter()
    for index, fold in enumerate(manifest.outer_folds):
        label = f"outer fold {index}"
        if type(fold.index) is not int or fold.index != index:
            raise ValueError(f"{label} has an invalid index")
        outer_fit = _validate_role(fold.fit_group_ids, positions, bin_counts, f"{label} fit")
        outer_test = _validate_role(fold.held_out_group_ids, positions, bin_counts, f"{label} holdout")
        if outer_fit & outer_test or outer_fit | outer_test != positions.keys():
            raise ValueError(f"{label} group membership must partition the view")
        outer_coverage.update(outer_test)

        if len(fold.inner_folds) != inner_count:
            raise ValueError(f"{label} must contain {inner_count} inner folds")
        inner_coverage: Counter[int] = Counter()
        for inner_index, inner in enumerate(fold.inner_folds):
            inner_label = f"{label}, inner fold {inner_index}"
            if type(inner.index) is not int or inner.index != inner_index:
                raise ValueError(f"{inner_label} has an invalid index")
            inner_fit = _validate_role(inner.fit_group_ids, positions, bin_counts, f"{inner_label} fit")
            validation = _validate_role(
                inner.validation_group_ids, positions, bin_counts, f"{inner_label} validation"
            )
            if inner_fit & validation or inner_fit | validation != outer_fit:
                raise ValueError(f"{inner_label} group membership must partition the outer fit groups")
            inner_coverage.update(validation)
        if not chrono and inner_coverage != Counter(outer_fit):
            raise ValueError(f"{label} inner validation coverage must include each fit group exactly once")

    if chrono:
        if manifest.outer_folds != (_chrono_fold(group_ids),):
            raise ValueError("manifest does not match the chronological prefix and suffix boundaries")
    elif outer_coverage != Counter(group_ids):
        raise ValueError("outer holdout coverage must include each view group exactly once")


def _validate_role(
    ids: tuple[int, ...], positions: dict[int, int], bin_counts: dict[int, int], label: str
) -> set[int]:
    if not ids:
        raise ValueError(f"{label} is empty")
    if any(type(group_id) is not int for group_id in ids):
        raise ValueError(f"{label} group IDs must be integers")
    membership = set(ids)
    if len(membership) != len(ids):
        raise ValueError(f"{label} contains duplicate group IDs")
    if membership - positions.keys():
        raise ValueError(f"{label} contains unknown group IDs")
    if tuple(sorted(ids, key=positions.__getitem__)) != ids:
        raise ValueError(f"{label} groups must follow chronological view order")
    if not any(bin_counts[group_id] for group_id in ids):
        raise ValueError(f"{label} has no usable bins in the supplied view")
    return membership


def _view_identity(view: ViewPlan) -> str:
    """Validate raw ownership and record metadata without accessing signal values."""
    meta = view.meta
    timestamps = meta.native_timestamps
    if not np.isfinite(timestamps).all() or not np.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("native timestamps must be in strict chronological order")
    timestamp_dtype = timestamps.dtype.newbyteorder("<")
    ticks = np.ascontiguousarray(timestamps, dtype=timestamp_dtype)
    timestamp_hash = _timestamp_hash(timestamps, ticks, timestamp_dtype)

    ids = view.group_ids
    if any(type(group_id) is not int or group_id < 0 for group_id in ids):
        raise ValueError("view group IDs must be nonnegative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("view contains duplicate group IDs")
    groups = []
    previous = None
    for group in view.groups:
        owned = group.owned_span
        if previous is not None:
            if owned.start < previous.start:
                raise ValueError("view group raw spans must follow chronological order")
            if owned.start < previous.stop:
                raise ValueError("view groups have overlapping raw spans")
        if owned.stop > len(timestamps):
            raise ValueError(f"group {group.group_id} raw span exceeds the native timebase")
        previous = owned
        blocks = []
        for block in group.blocks:
            bins = sha256()
            for span in block.spans:
                if span.start < owned.start or span.stop > owned.stop:
                    raise ValueError(f"planned bin raw span escapes group {group.group_id} ownership")
                bins.update(_RAW_BOUNDS.pack(span.start, span.stop))
            blocks.append({
                "trial_id": block.trial_id,
                "n_bins": len(block.spans),
                "raw_spans_sha256": bins.hexdigest(),
            })
        groups.append({
            "group_id": group.group_id,
            "trial_ids": group.trial_ids,
            "eligible_trial_ids": group.eligible_trial_ids,
            "owned_span": asdict(owned),
            "blocks": blocks,
        })
    return _canonical_json({
        "session": {
            "dataset": meta.dataset,
            "session_id": meta.session_id,
            "revision": meta.revision,
            "native_bin_width_s": float(meta.native_bin_width_s),
            "n_native_samples": len(timestamps),
            "timestamp_dtype": timestamp_dtype.str,
            "timestamps_sha256": timestamp_hash,
            "unit_ids": meta.unit_ids,
            "target_field": meta.target_field,
            "target_units": meta.target_units,
        },
        "mode": view.mode,
        "bin_width_s": float(view.bin_width_s),
        "groups": groups,
        "excluded_trial_ids": view.excluded_trial_ids,
        "dropped_spans": [asdict(span) for span in view.dropped_spans],
    })


def _timestamp_hash(timestamps: np.ndarray, ticks: np.ndarray, dtype: np.dtype) -> str:
    """Hash timestamp values without including x86 long-double padding bytes."""
    if dtype.kind == "f" and dtype.itemsize == 16 and np.finfo(dtype).nmant == 63:
        digest = sha256()
        for value in timestamps:
            canonical = np.format_float_scientific(value, unique=True, trim="k")
            digest.update(canonical.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    return sha256(ticks).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field, value in pairs:
        if field in result:
            raise ValueError(f"duplicate split manifest field: {field!r}")
        result[field] = value
    return result


def _record(value: object, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or value.keys() != fields:
        raise ValueError(f"invalid {label} fields")
    return value


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _read_outer_fold(value: object) -> OuterFold:
    fold = _record(value, {"index", "fit_group_ids", "held_out_group_ids", "inner_folds"}, "outer fold")
    return OuterFold(
        fold["index"],
        tuple(_array(fold["fit_group_ids"], "fit group IDs")),
        tuple(_array(fold["held_out_group_ids"], "held-out group IDs")),
        tuple(_read_inner_fold(inner) for inner in _array(fold["inner_folds"], "inner folds")),
    )


def _read_inner_fold(value: object) -> InnerFold:
    fold = _record(value, {"index", "fit_group_ids", "validation_group_ids"}, "inner fold")
    return InnerFold(
        fold["index"],
        tuple(_array(fold["fit_group_ids"], "fit group IDs")),
        tuple(_array(fold["validation_group_ids"], "validation group IDs")),
    )
