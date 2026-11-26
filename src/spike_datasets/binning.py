"""Plan split ownership from metadata and bin selected native signal spans."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass
import math
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .types import BinnedBlock, RawSpan, SessionMeta, TrialMeta

ViewMode = Literal["trial", "fixed"]


class _NativeBlock(Protocol):
    @property
    def counts(self) -> NDArray[np.number]: ...

    @property
    def targets(self) -> NDArray[np.number] | None: ...


class SpanReader(Protocol):
    """Read time-major native counts and, when requested, behavior for one span."""

    def __call__(self, span: RawSpan, /, *, with_behavior: bool) -> _NativeBlock: ...


@dataclass(frozen=True)
class PlannedBlock:
    """Consecutive complete bins from a trial or from the session grid."""

    trial: TrialMeta | None
    spans: tuple[RawSpan, ...]

    @property
    def trial_id(self) -> int | None:
        return None if self.trial is None else self.trial.row_id


@dataclass(frozen=True)
class ViewGroup:
    """Trials and native ownership shared by both views, with view-specific bins."""

    group_id: int
    trial_ids: tuple[int, ...]
    eligible_trial_ids: tuple[int, ...]
    owned_span: RawSpan
    blocks: tuple[PlannedBlock, ...]

    @property
    def n_bins(self) -> int:
        return sum(len(block.spans) for block in self.blocks)


@dataclass(frozen=True)
class ViewPlan:
    """Metadata-only binning plan with groups in native time order.

    Excluded trial IDs include malformed source rows and trials with no complete bin.
    """

    meta: SessionMeta
    mode: ViewMode
    bin_width_s: float
    groups: tuple[ViewGroup, ...]
    excluded_trial_ids: tuple[int, ...]
    dropped_spans: tuple[RawSpan, ...]

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(group.group_id for group in self.groups)


@dataclass(frozen=True)
class ViewBlock(BinnedBlock):
    """Binned signals with their group ID and original trial, when applicable."""

    group_id: int
    trial: TrialMeta | None

    @property
    def trial_id(self) -> int | None:
        return None if self.trial is None else self.trial.row_id


@dataclass(frozen=True)
class ViewData:
    """Materialized blocks in plan order with the session's channel and target metadata."""

    meta: SessionMeta
    mode: ViewMode
    bin_width_s: float
    group_ids: tuple[int, ...]
    blocks: tuple[ViewBlock, ...]


@dataclass(frozen=True)
class _TrialComponent:
    start: int
    trial_ids: tuple[int, ...]
    eligible_trials: tuple[tuple[TrialMeta, RawSpan], ...]


def plan_view(
    meta: SessionMeta, *, mode: ViewMode, bin_width_s: float = 0.02
) -> ViewPlan:
    """Plan bins and their common split groups without opening signal arrays."""
    if mode not in ("trial", "fixed"):
        raise ValueError(f"unsupported view mode: {mode!r}")
    width = _bin_samples(meta, bin_width_s)
    components, excluded, trial_dropped = _plan_trial_components(meta, width)
    if not components:
        raise ValueError("no recorded trial contains a complete trial bin")

    n_samples = len(meta.native_timestamps)
    # Each retained group owns the gap up to the next retained group.
    boundaries = [0, *(component.start for component in components[1:]), n_samples]
    groups = []
    for group_id, component in enumerate(components):
        owned_span = RawSpan(boundaries[group_id], boundaries[group_id + 1])
        if mode == "trial":
            blocks = tuple(
                PlannedBlock(trial, _complete_bins(span.start, span.stop, width))
                for trial, span in component.eligible_trials
            )
        else:
            start = ((owned_span.start + width - 1) // width) * width
            spans = _complete_bins(start, owned_span.stop, width)
            blocks = (PlannedBlock(None, spans),) if spans else ()
        groups.append(
            ViewGroup(
                group_id=group_id,
                trial_ids=component.trial_ids,
                eligible_trial_ids=tuple(trial.row_id for trial, _ in component.eligible_trials),
                owned_span=owned_span,
                blocks=blocks,
            )
        )

    dropped = (
        trial_dropped
        if mode == "trial"
        else _fixed_dropped(boundaries[1:-1], n_samples, width)
    )
    return ViewPlan(meta, mode, bin_width_s, tuple(groups), excluded, dropped)


def _bin_samples(meta: SessionMeta, bin_width_s: float) -> int:
    timestamps = meta.native_timestamps
    if not len(timestamps) or timestamps.dtype.kind not in "iuf":
        raise ValueError("native timebase must contain real numeric timestamps")
    if not np.isfinite(timestamps).all():
        raise ValueError("native timebase must contain finite timestamps")
    numeric_dtype = np.result_type(timestamps.dtype, np.float64)
    values = timestamps.astype(numeric_dtype, copy=False)
    native_width = meta.native_bin_width_s
    tolerance = native_width * 1e-6
    step_tolerance = grid_tolerance = tolerance
    if timestamps.dtype.kind == "f" and len(values) > 1:
        # Each stored endpoint can differ from its true time by half a representable step.
        with np.errstate(over="ignore"):
            spacing = np.spacing(np.abs(timestamps)).astype(numeric_dtype, copy=False)
        if spacing.max() + tolerance >= native_width / 2:
            raise ValueError("native timestamp precision is too coarse to validate a uniform timebase")
        endpoint_error = spacing * 0.5
        step_tolerance = tolerance + endpoint_error[:-1] + endpoint_error[1:]
        grid_tolerance = tolerance + endpoint_error[0] + endpoint_error[1:]

    steps = np.diff(values)
    elapsed = values[1:] - values[0]
    nominal = np.arange(1, len(values), dtype=numeric_dtype) * native_width
    if (
        np.any(steps <= 0)
        or not np.all(np.abs(steps - native_width) <= step_tolerance)
        or not np.all(np.abs(elapsed - nominal) <= grid_tolerance)
    ):
        raise ValueError("native timebase must be uniform and match the native bin width")

    if (
        isinstance(bin_width_s, (bool, np.bool_))
        or not math.isfinite(bin_width_s)
        or bin_width_s <= 0
    ):
        raise ValueError("bin width must be positive and finite")
    ratio = float(bin_width_s) / float(meta.native_bin_width_s)
    if not math.isfinite(ratio):
        raise ValueError("bin width must be a finite number of native samples")
    width = round(ratio)
    if width < 1 or not math.isclose(ratio, width, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("bin width must be a whole number of native samples")
    return width


def _trial_components(trials: tuple[TrialMeta, ...]) -> Iterator[list[TrialMeta]]:
    component: list[TrialMeta] = []
    stop = -math.inf
    for trial in sorted(trials, key=lambda trial: (trial.start_s, trial.row_id)):
        if component and trial.start_s >= stop:
            yield component
            component = []
        component.append(trial)
        stop = max(stop, trial.end_s)
    if component:
        yield component


def _plan_trial_components(
    meta: SessionMeta, width: int
) -> tuple[list[_TrialComponent], tuple[int, ...], tuple[RawSpan, ...]]:
    timestamps = meta.native_timestamps
    ids = {trial.row_id for trial in meta.trials}
    if len(ids) != len(meta.trials):
        raise ValueError("trial row IDs must be unique")
    components = []
    excluded = list(meta.invalid_trial_ids)
    dropped = []
    for trials in _trial_components(meta.trials):
        bounds = np.searchsorted(
            timestamps,
            [(trial.start_s, trial.end_s) for trial in trials],
            side="left",
        )
        eligible_trials = []
        for trial, (start, stop) in zip(trials, bounds):
            start, stop = int(start), int(stop)
            complete_stop = stop - (stop - start) % width
            if start < complete_stop:
                eligible_trials.append((trial, RawSpan(start, complete_stop)))
            else:
                excluded.append(trial.row_id)
            if complete_stop < stop:
                dropped.append(RawSpan(complete_stop, stop))
        if eligible_trials:
            components.append(
                _TrialComponent(
                    int(bounds[0, 0]),
                    tuple(trial.row_id for trial in trials),
                    tuple(eligible_trials),
                )
            )
    return components, tuple(sorted(excluded)), tuple(sorted(dropped))


def _complete_bins(start: int, stop: int, width: int) -> tuple[RawSpan, ...]:
    return tuple(
        RawSpan(index, index + width) for index in range(start, stop - width + 1, width)
    )


def _fixed_dropped(boundaries: list[int], n_samples: int, width: int) -> tuple[RawSpan, ...]:
    tail_start = n_samples - n_samples % width
    crossing_starts = {
        boundary - boundary % width
        for boundary in boundaries
        if boundary % width and boundary < tail_start
    }
    spans = [RawSpan(start, start + width) for start in sorted(crossing_starts)]
    if tail_start < n_samples:
        spans.append(RawSpan(tail_start, n_samples))
    return tuple(spans)


def materialize(
    view: ViewPlan, group_ids: Collection[int], read_span: SpanReader, *, with_behavior: bool
) -> ViewData:
    """Read only complete bins of the selected groups, in plan order."""
    if any(
        isinstance(group_id, (bool, np.bool_)) or not isinstance(group_id, (int, np.integer))
        for group_id in group_ids
    ):
        raise ValueError("group IDs must be integers")
    selected = set(group_ids)
    if len(selected) != len(group_ids) or not selected.issubset(view.group_ids):
        raise ValueError("group IDs must be unique and present in the view plan")
    groups = tuple(group for group in view.groups if group.group_id in selected)
    if not any(group.blocks for group in groups):
        raise ValueError("the selected groups have no bins in this view")

    blocks = tuple(
        _read_block(view.meta, group.group_id, block, read_span, with_behavior=with_behavior)
        for group in groups
        for block in group.blocks
    )
    return ViewData(
        view.meta, view.mode, view.bin_width_s, tuple(group.group_id for group in groups), blocks
    )


def _read_block(
    meta: SessionMeta,
    group_id: int,
    block: PlannedBlock,
    read_span: SpanReader,
    *,
    with_behavior: bool,
) -> ViewBlock:
    first, last = block.spans[0], block.spans[-1]
    span = RawSpan(first.start, last.stop)
    native = read_span(span, with_behavior=with_behavior)
    n_bins = len(block.spans)
    width = first.stop - first.start
    counts = _sum_counts(native.counts, n_bins, width, len(meta.unit_ids))

    targets = target_mask = None
    if with_behavior:
        behavior = native.targets
        if (
            behavior is None
            or behavior.ndim != 2
            or behavior.shape[0] != n_bins * width
            or behavior.dtype.kind not in "iuf"
        ):
            raise ValueError("behavior must be a real-valued matrix with one row per native sample")
        values = behavior.reshape(n_bins, width, behavior.shape[1])
        target_mask = np.isfinite(values).all(axis=1)
        values = np.where(target_mask[:, None, :], values, 0).astype(
            np.result_type(behavior.dtype, np.float64), copy=False
        )
        with np.errstate(over="ignore", invalid="ignore"):
            targets = values.mean(axis=1)
        overflow = ~np.isfinite(targets)
        if overflow.any():
            # Rescale exceptional axes so the intermediate sum stays finite.
            rows = values.swapaxes(1, 2)[overflow]
            scale = np.abs(rows).max(axis=1)
            targets[overflow] = (rows / scale[:, None]).mean(axis=1) * scale
        targets[~target_mask] = np.nan

    timestamps = meta.native_timestamps[[span.start for span in block.spans]]
    return ViewBlock(counts, targets, target_mask, block.spans, timestamps, group_id, block.trial)


def _sum_counts(
    counts: NDArray[np.number], n_bins: int, width: int, n_units: int
) -> NDArray[np.int64]:
    if counts.shape != (n_bins * width, n_units):
        raise ValueError("spike counts must have one row per native sample and one column per unit")
    if counts.dtype.kind not in "iuf" or np.any(counts < 0):
        raise ValueError("spike counts must be finite, nonnegative integers")
    if counts.dtype.kind == "f" and (
        not np.isfinite(counts).all() or np.any(counts != np.floor(counts))
    ):
        raise ValueError("spike counts must be finite, nonnegative integers")

    largest_count = int(counts.max(initial=0))
    limit = np.iinfo(np.int64).max
    if largest_count > limit:
        raise ValueError("spike counts exceed the int64 range")
    values = counts.astype(np.int64, copy=False).reshape(n_bins, width, n_units)
    if largest_count <= limit // width:
        return values.sum(axis=1, dtype=np.int64)

    totals = values.sum(axis=1, dtype=object)
    if np.any(totals > limit):
        raise ValueError("binned spike counts exceed the int64 range")
    return totals.astype(np.int64)
