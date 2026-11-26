"""Shared value types passed between readers, binners, and noise handling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Dataset = Literal["chewie", "jango"]


@dataclass(frozen=True)
class Source:
    """Identity and location of one source recording."""

    path: Path
    dataset: Dataset
    session_id: str
    revision: str

    def __post_init__(self) -> None:
        if self.dataset not in ("chewie", "jango"):
            raise ValueError(f"unsupported dataset: {self.dataset!r}")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.revision:
            raise ValueError("revision must not be empty")


@dataclass(frozen=True)
class TrialMeta:
    """Original trial row and its time bounds and optional annotations."""

    row_id: int
    start_s: float
    end_s: float
    cue: str | None = None
    outcome: str | None = None
    direction: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.row_id, bool) or not isinstance(self.row_id, int):
            raise TypeError("row_id must be an integer")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("trial bounds must be finite")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("trial bounds must be nonnegative and nonempty")


@dataclass(frozen=True)
class SessionMeta:
    """Metadata needed to plan views without reading signal values."""

    dataset: Dataset
    session_id: str
    revision: str
    native_bin_width_s: float
    native_timestamps: NDArray[np.floating]
    unit_ids: tuple[str, ...]
    target_field: str
    target_units: str
    trials: tuple[TrialMeta, ...]

    def __post_init__(self) -> None:
        if self.dataset not in ("chewie", "jango"):
            raise ValueError(f"unsupported dataset: {self.dataset!r}")
        if not self.session_id or not self.revision:
            raise ValueError("session_id and revision must not be empty")
        if not math.isfinite(self.native_bin_width_s) or self.native_bin_width_s <= 0:
            raise ValueError("native_bin_width_s must be positive and finite")
        if self.native_timestamps.ndim != 1 or not np.issubdtype(
            self.native_timestamps.dtype, np.number
        ):
            raise ValueError("native_timestamps must be a one-dimensional numeric array")
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise ValueError("unit_ids must be unique and preserve channel identity")
        if not self.target_field:
            raise ValueError("target_field must not be empty")
        if not isinstance(self.trials, tuple):
            raise TypeError("trials must be an ordered tuple of TrialMeta records")


@dataclass(frozen=True, order=True)
class RawSpan:
    """Half-open native sample indices ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("stop", self.stop)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer native index")
        if self.start < 0:
            raise ValueError("start must be nonnegative")
        if self.stop <= self.start:
            raise ValueError("RawSpan must have nonempty, forward half-open bounds")


@dataclass(frozen=True)
class BinnedBlock:
    """Binned neural counts and optional behavior aligned on the sample axis.

    ``counts`` has shape ``(samples, units)`` and keeps its integer dtype.
    When present, ``targets`` and ``target_mask`` have shape ``(samples, axes)``;
    the mask is true only where the corresponding target is finite. ``spans``
    and ``timestamps`` each have one entry per sample and retain source order.
    """

    counts: NDArray[np.integer]
    targets: NDArray[np.number] | None
    target_mask: NDArray[np.bool_] | None
    spans: tuple[RawSpan, ...]
    timestamps: NDArray[np.number]

    def __post_init__(self) -> None:
        if self.counts.ndim != 2 or not np.issubdtype(self.counts.dtype, np.integer):
            raise ValueError("counts must be a two-dimensional integer array")
        if np.any(self.counts < 0):
            raise ValueError("counts must be nonnegative")
        n_samples = self.counts.shape[0]
        if self.timestamps.ndim != 1 or self.timestamps.shape[0] != n_samples:
            raise ValueError("timestamps must have one entry per count row")
        if not np.issubdtype(self.timestamps.dtype, np.number):
            raise ValueError("timestamps must have a numeric dtype")
        if len(self.spans) != n_samples or not all(
            isinstance(span, RawSpan) for span in self.spans
        ):
            raise ValueError("spans must contain one RawSpan per count row")
        if (self.targets is None) != (self.target_mask is None):
            raise ValueError("targets and target_mask must either both be present or both absent")
        if self.targets is not None:
            if self.targets.ndim != 2 or self.targets.shape[0] != n_samples:
                raise ValueError("targets must have one row per count row")
            if not np.issubdtype(self.targets.dtype, np.number):
                raise ValueError("targets must have a numeric dtype")
            if self.target_mask is None or self.target_mask.shape != self.targets.shape:
                raise ValueError("target_mask must match the targets shape")
            if self.target_mask.dtype != np.dtype(bool):
                raise ValueError("target_mask must have boolean dtype")
            if np.any(self.target_mask & ~np.isfinite(self.targets)):
                raise ValueError("target_mask cannot mark a nonfinite target as valid")
