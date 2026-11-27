"""Nested development, refit, and held-out scoring for one outer fold."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .binning import ViewData, ViewPlan, materialize
from .noise import NoiseState, augment_fit, fit_noise
from .splits import OuterFold, SplitManifest
from .types import RawSpan, Source
from .xds import read_span

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class CleanData:
    """Clean counts and optional behavior for one development or scoring role."""

    clean_counts: NDArray[np.integer]
    targets: NDArray[np.number] | None
    target_mask: NDArray[np.bool_] | None
    group_ids: tuple[int, ...]
    spans: tuple[RawSpan, ...]
    timestamps: NDArray[np.number]


@dataclass(frozen=True)
class FitData(CleanData):
    """Clean count observations plus one deterministic augmented feature view."""

    features: NDArray[np.floating]
    noise_state: NoiseState


@dataclass(frozen=True)
class FitAndValidation:
    fit: FitData
    validation: CleanData


@dataclass(frozen=True)
class FrozenSelection:
    """Immutable settings tied to exactly one source, manifest, and outer fold."""

    source_id: tuple[str, str, str]
    manifest_id: str
    fold_index: int
    settings_json: str

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(json.loads(self.settings_json))


@dataclass(frozen=True)
class Score:
    """Scorer result and the identity/settings under which it was produced."""

    value: Any
    settings: Mapping[str, JsonValue]
    source_id: tuple[str, str, str]
    manifest_id: str
    fold_index: int
    n_samples: int


@dataclass(frozen=True)
class Experiment:
    source: Source
    view: ViewPlan
    manifest: SplitManifest
    outer_fold: OuterFold

    def inner_data(
        self,
        index: int,
        *,
        noise_factor: float = 0.02,
        noise_seed: int,
        with_behavior: bool = True,
    ) -> FitAndValidation:
        """Materialize inner-fit and validation roles; fit augmentation on fit only."""
        inner = self._inner_fold(index)
        fit_view = materialize(self.view, inner.fit_group_ids, self._read, with_behavior=with_behavior)
        validation_view = materialize(
            self.view, inner.validation_group_ids, self._read, with_behavior=with_behavior
        )
        fit = _fit_data(fit_view, noise_factor, noise_seed)
        return FitAndValidation(fit, _clean_data(validation_view))

    def lock_selection(self, settings: Mapping[str, JsonValue]) -> FrozenSelection:
        """Freeze JSON-compatible selected settings, including the noise factor."""
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        values = dict(settings)
        if "noise_factor" not in values:
            raise ValueError("locked settings must include noise_factor")
        factor = values["noise_factor"]
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ValueError("noise_factor must be a finite number from 0.0 to 0.05")
        try:
            fit_noise([np.zeros((1, len(self.view.meta.unit_ids)), dtype=np.int64)], factor=float(factor))
        except ValueError as error:
            raise ValueError("noise_factor must be a finite number from 0.0 to 0.05") from error
        try:
            encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("settings must contain only finite JSON values") from error
        # Round trip normalizes tuples and mapping subclasses to canonical JSON values.
        normalized = json.loads(encoded)
        return FrozenSelection(_source_id(self.source), self.manifest.manifest_id,
                               self.outer_fold.index, json.dumps(normalized, sort_keys=True,
                                                                  separators=(",", ":")))

    def refit_data(
        self,
        selection: FrozenSelection,
        *,
        noise_seed: int,
        with_behavior: bool = True,
    ) -> FitData:
        """Read all outer held-in bins and fit noise on those clean counts."""
        self._validate_selection(selection)
        data = materialize(self.view, self.outer_fold.fit_group_ids, self._read,
                           with_behavior=with_behavior)
        return _fit_data(data, float(selection.settings["noise_factor"]), noise_seed)

    def score(
        self,
        selection: FrozenSelection,
        scorer: Callable[[CleanData], Any],
        *,
        with_behavior: bool = True,
    ) -> Score:
        """Call a frozen scorer with clean held-out inputs only."""
        self._validate_selection(selection)
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        data = materialize(self.view, self.outer_fold.held_out_group_ids, self._read,
                           with_behavior=with_behavior)
        clean = _clean_data(data)
        result = scorer(clean)
        return Score(result, selection.settings, _source_id(self.source),
                     self.manifest.manifest_id, self.outer_fold.index,
                     clean.clean_counts.shape[0])

    def _read(self, span: RawSpan, *, with_behavior: bool):
        return read_span(self.source, span, with_behavior=with_behavior)

    def _inner_fold(self, index: int):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("inner fold index must be an integer")
        if index < 0 or index >= len(self.outer_fold.inner_folds):
            raise ValueError(f"unknown inner fold index: {index}")
        return self.outer_fold.inner_folds[index]

    def _validate_selection(self, selection: FrozenSelection) -> None:
        if not isinstance(selection, FrozenSelection):
            raise TypeError("selection must be a FrozenSelection")
        if selection.source_id != _source_id(self.source):
            raise ValueError("selection source does not match this experiment")
        if selection.manifest_id != self.manifest.manifest_id:
            raise ValueError("selection manifest does not match this experiment")
        if selection.fold_index != self.outer_fold.index:
            raise ValueError("selection fold does not match this experiment")


def open_experiment(
    source: Source, view: ViewPlan, manifest: SplitManifest, outer_fold: int
) -> Experiment:
    """Bind one metadata-only view and validated split manifest to an outer fold."""
    if (source.dataset, source.session_id, source.revision) != (
        view.meta.dataset, view.meta.session_id, view.meta.revision
    ):
        raise ValueError("source identity does not match the view metadata")
    # Replaying validates manifest identity, group membership, and role coverage.
    from .splits import load_manifest
    load_manifest(manifest.to_json(), view)
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int):
        raise ValueError("outer fold index must be an integer")
    if outer_fold < 0 or outer_fold >= len(manifest.outer_folds):
        raise ValueError(f"unknown outer fold index: {outer_fold}")
    return Experiment(source, view, manifest, manifest.outer_folds[outer_fold])


def _source_id(source: Source) -> tuple[str, str, str]:
    return source.dataset, source.session_id, source.revision


def _clean_data(data: ViewData) -> CleanData:
    blocks = data.blocks
    return CleanData(
        clean_counts=_concat([block.counts for block in blocks], (0, len(data.meta.unit_ids)), np.int64),
        targets=_concat_optional([block.targets for block in blocks]),
        target_mask=_concat_optional([block.target_mask for block in blocks]),
        group_ids=tuple(block.group_id for block in blocks for _ in block.spans),
        spans=tuple(span for block in blocks for span in block.spans),
        timestamps=np.concatenate([block.timestamps for block in blocks]) if blocks else np.empty(0),
    )


def _fit_data(data: ViewData, factor: float, seed: int) -> FitData:
    clean = _clean_data(data)
    state = fit_noise([clean.clean_counts], factor=factor)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("noise_seed must be a nonnegative integer")
    features = augment_fit(clean.clean_counts, state, rng=np.random.default_rng(seed))
    return FitData(**clean.__dict__, features=features, noise_state=state)


def _concat(arrays, empty_shape, dtype=None):
    nonempty = [array for array in arrays if array is not None]
    if not nonempty:
        return np.empty(empty_shape, dtype=dtype)
    return np.concatenate(nonempty, axis=0)


def _concat_optional(arrays):
    if not arrays or arrays[0] is None:
        return None
    return np.concatenate(arrays, axis=0)
