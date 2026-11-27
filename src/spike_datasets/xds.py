"""Metadata inspection and bounded signal reads for MATLAB v7.3 XDS files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
from numpy.typing import NDArray

from .binning import _validated_native_width
from .types import Dataset, RawSpan, SessionMeta, Source, TrialMeta


@dataclass(frozen=True)
class RawBlock:
    """Native spike counts and optional behavior, with time on the first axis."""

    counts: NDArray[np.number]
    targets: NDArray[np.number] | None


@dataclass(frozen=True)
class _DatasetSpec:
    subject: str
    release_year: int
    target_field: str


_SPECS: dict[Dataset, _DatasetSpec] = {
    "chewie": _DatasetSpec("Chewie", 2016, "curs_v"),
    "jango": _DatasetSpec("Jango", 2015, "force"),
}


def discover_sessions(root: Path, dataset: Dataset, *, revision: str) -> tuple[Source, ...]:
    """List matching files in the extracted release directory in recording order."""
    if dataset not in _SPECS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    spec = _SPECS[dataset]
    pattern = re.compile(rf"{spec.subject}_({spec.release_year}[0-9]{{4}})_([0-9]+)\.mat")
    recordings = []
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            recordings.append((match[1], int(match[2]), path))
    return tuple(
        Source(path=path, dataset=dataset, session_id=path.stem, revision=revision)
        for _, _, path in sorted(recordings)
    )


def inspect_session(source: Source) -> SessionMeta:
    """Read planning metadata and check signal shapes without reading signal values."""
    spec = _SPECS[source.dataset]
    with h5py.File(source.path, "r") as file:
        xds = _xds_group(file, source)
        time_field = _field(xds, source, "time_frame")
        n_samples = _time_size(time_field, source)
        timestamps = time_field[...].reshape(-1)
        width = _native_width(xds, source, timestamps)
        unit_field = _field(xds, source, "unit_names")
        n_units = _unit_count(unit_field, source)
        spikes = _field(xds, source, "spike_counts")
        _time_axis(spikes, source, n_samples, n_units)
        target = _field(xds, source, spec.target_field)
        _time_axis(target, source, n_samples, 2)
        trial_meta, invalid_trial_ids = _trials(xds, source)
        unit_ids = _unit_ids(unit_field, source)
        target_units = target.attrs.get("units", "recorded XDS units")
        if isinstance(target_units, bytes):
            target_units = target_units.decode("utf-8")
        if not isinstance(target_units, str) or not target_units.strip():
            target_units = "recorded XDS units"

    return SessionMeta(
        dataset=source.dataset,
        session_id=source.session_id,
        revision=source.revision,
        native_bin_width_s=width,
        native_timestamps=timestamps,
        unit_ids=unit_ids,
        target_field=spec.target_field,
        target_units=target_units,
        trials=trial_meta,
        invalid_trial_ids=invalid_trial_ids,
    )


def read_span(source: Source, span: RawSpan, *, with_behavior: bool) -> RawBlock:
    """Read the requested half-open native sample range in source channel order."""
    with h5py.File(source.path, "r") as file:
        xds = _xds_group(file, source)
        n_samples = _time_size(_field(xds, source, "time_frame"), source)
        if span.stop > n_samples:
            raise _error(source, "span", f"[{span.start}, {span.stop}) exceeds {n_samples} samples")
        n_units = _unit_count(_field(xds, source, "unit_names"), source)
        spikes = _field(xds, source, "spike_counts")
        spike_axis = _time_axis(spikes, source, n_samples, n_units)
        targets = None
        if with_behavior:
            target = _field(xds, source, _SPECS[source.dataset].target_field)
            target_axis = _time_axis(target, source, n_samples, 2)
        counts = _read_rows(spikes, spike_axis, span)
        if with_behavior:
            targets = _read_rows(target, target_axis, span)
    return RawBlock(counts=counts, targets=targets)


def _error(source: Source, field: str, message: str) -> ValueError:
    return ValueError(f"{source.session_id} ({source.path}): {field} {message}")


def _xds_group(file: h5py.File, source: Source) -> h5py.Group:
    xds = file.get("xds")
    if not isinstance(xds, h5py.Group):
        raise _error(source, "xds", "must be an HDF5 group")
    return xds


def _field(xds: h5py.Group, source: Source, name: str) -> h5py.Dataset:
    field = xds.get(name)
    if not isinstance(field, h5py.Dataset):
        raise _error(source, name, "must be present as an HDF5 dataset")
    return field


def _vector_size(field: h5py.Dataset, source: Source) -> int:
    if field.shape is None or (
        field.ndim != 1 and (field.ndim != 2 or (1 not in field.shape and field.size != 0))
    ):
        raise _error(source, field.name, "must be a vector")
    return field.size


def _time_size(field: h5py.Dataset, source: Source) -> int:
    size = _vector_size(field, source)
    if size == 0 or field.dtype.kind not in "iuf":
        raise _error(source, field.name, "must be a nonempty numeric timestamp vector")
    return size


def _native_width(
    xds: h5py.Group, source: Source, timestamps: NDArray[np.number]
) -> float:
    field = _field(xds, source, "bin_width")
    if field.size != 1 or field.dtype.kind not in "iuf":
        raise _error(source, field.name, "must be a numeric scalar")
    # Preserve the HDF5 scalar precision for cadence and whole-bin validation.
    width = field[...].reshape(-1)[0]
    if not np.isfinite(width) or width <= 0:
        raise _error(source, field.name, "must be finite and positive")
    try:
        return _validated_native_width(timestamps, width)
    except ValueError as exc:
        raise _error(source, "time_frame", str(exc)) from exc


def _unit_count(field: h5py.Dataset, source: Source) -> int:
    size = _vector_size(field, source)
    if size == 0 or h5py.check_dtype(ref=field.dtype) is not h5py.Reference:
        raise _error(source, field.name, "must be a nonempty vector of MATLAB cell references")
    return size


def _unit_ids(field: h5py.Dataset, source: Source) -> tuple[str, ...]:
    ids = []
    file = field.file
    for index, reference in enumerate(field[...].reshape(-1)):
        message = f"contains an invalid character reference at row {index}"
        try:
            text = file[reference]
        except (KeyError, ValueError, RuntimeError) as exc:
            raise _error(source, field.name, message) from exc
        if not isinstance(text, h5py.Dataset) or text.attrs.get("MATLAB_class") not in (
            "char",
            b"char",
        ):
            raise _error(source, field.name, message)
        try:
            label = _matlab_text(text, source).rstrip("\x00")
        except ValueError as exc:
            raise _error(source, field.name, message) from exc
        if not label:
            raise _error(source, field.name, message)
        ids.append(label)
    return tuple(ids)


def _matlab_text(field: h5py.Dataset, source: Source) -> str:
    _vector_size(field, source)
    if field.dtype.kind != "u" or field.dtype.itemsize != 2:
        raise _error(source, field.name, "must contain UTF-16 MATLAB character data")
    codes = field[...].astype("<u2", copy=False)
    try:
        return codes.tobytes().decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise _error(source, field.name, "contains invalid UTF-16 character data") from exc


def _time_axis(
    field: h5py.Dataset, source: Source, n_samples: int, n_channels: int
) -> int:
    if field.dtype.kind not in "iuf":
        raise _error(source, field.name, "must contain numeric signal values")
    if field.shape == (n_channels, n_samples):
        # MATLAB v7.3 stores the channel axis first, including square matrices.
        return 1
    if field.shape == (n_samples, n_channels):
        return 0
    raise _error(
        source,
        field.name,
        f"must align {n_channels} channels with {n_samples} time_frame samples, got {field.shape}",
    )


def _read_rows(field: h5py.Dataset, time_axis: int, span: RawSpan) -> NDArray[np.number]:
    rows = slice(span.start, span.stop)
    if time_axis == 1:
        return field[:, rows].T
    return field[rows, :]


def _trials(
    xds: h5py.Group, source: Source
) -> tuple[tuple[TrialMeta, ...], tuple[int, ...]]:
    start_field = _field(xds, source, "trial_start_time")
    end_field = _field(xds, source, "trial_end_time")
    n_trials = _vector_size(start_field, source)
    if _vector_size(end_field, source) != n_trials:
        raise _error(source, end_field.name, "must have one entry per trial_start_time row")
    for field in (start_field, end_field):
        if field.dtype.kind not in "iuf":
            raise _error(source, field.name, "must contain numeric trial bounds")
    starts = start_field[...].reshape(-1)
    ends = end_field[...].reshape(-1)
    cues = _numeric_annotations(xds, source, "trial_gocue_time", n_trials)
    directions = _numeric_annotations(xds, source, "trial_target_dir", n_trials)
    outcomes = _outcomes(xds, source, n_trials)
    trials = []
    invalid_ids = []
    for row_id, (start, end) in enumerate(zip(starts, ends)):
        start_s, end_s = float(start), float(end)
        if not (np.isfinite(start_s) and np.isfinite(end_s) and 0 <= start_s < end_s):
            invalid_ids.append(row_id)
            continue
        trials.append(
            TrialMeta(
                row_id=row_id,
                start_s=start_s,
                end_s=end_s,
                cue=cues[row_id],
                outcome=outcomes[row_id],
                direction=directions[row_id],
            )
        )
    return tuple(trials), tuple(invalid_ids)


def _numeric_annotations(
    xds: h5py.Group, source: Source, name: str, n_trials: int
) -> list[str | None]:
    if name not in xds:
        return [None] * n_trials
    field = _field(xds, source, name)
    if field.dtype.kind not in "iuf":
        raise _error(source, field.name, "must contain numeric trial annotations")
    values = field[...]
    # XDS exporters may store zero as a scalar when a trial field is unavailable.
    if values.size == 0 or (n_trials != 1 and values.size == 1 and values.item() == 0):
        return [None] * n_trials
    if values.ndim <= 1 or (
        values.ndim == 2 and 1 in values.shape and values.size == n_trials
    ):
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_trials:
        raise _error(source, field.name, "must have one column per trial row")
    return [
        ", ".join(str(value) for value in row if np.isfinite(value)) or None for row in values.T
    ]


def _outcomes(xds: h5py.Group, source: Source, n_trials: int) -> list[str | None]:
    if "trial_result" not in xds:
        return [None] * n_trials
    field = _field(xds, source, "trial_result")
    outcomes = _matlab_text(field, source)
    if not outcomes:
        return [None] * n_trials
    if len(outcomes) != n_trials:
        raise _error(source, field.name, "must have one character per trial row")
    return [outcome if outcome != "\x00" else None for outcome in outcomes]
