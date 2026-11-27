"""Fixed, fold-matched Day-0 to Day-K experiment pairs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import re

from .evaluation import CleanData, Experiment, FitAndValidation, FitData, FrozenSelection, Score
from .evaluation import JsonValue


@dataclass(frozen=True)
class FixedPair:
    """Validated source and later target experiments for one compatible fold."""

    day0: Experiment
    dayk: Experiment

    def lock_zero_shot(self, settings: Mapping[str, JsonValue]) -> FrozenSelection:
        """Lock source-developed zero-shot choices without reading target signals."""
        return self.dayk.lock_selection(settings)

    def adaptation_data(
        self, index: int, *, noise_factor: float = 0.02, noise_seed: int
    ) -> FitAndValidation:
        """Read target held-in spikes for adaptation and keep target behavior closed."""
        return self.dayk.inner_data(index, noise_factor=noise_factor, noise_seed=noise_seed,
                                    with_behavior=False)

    def lock_adaptation(self, settings: Mapping[str, JsonValue]) -> FrozenSelection:
        """Freeze adaptation choices against the target pair identity."""
        return self.dayk.lock_selection(settings)

    def refit_adaptation(self, selection: FrozenSelection, *, noise_seed: int) -> FitData:
        """Refit an adapted model on all target held-in spikes, without behavior."""
        return self.dayk.refit_data(selection, noise_seed=noise_seed, with_behavior=False)

    def score_zero_shot(
        self, selection: FrozenSelection, scorer: Callable[[CleanData], Any]
    ) -> Score:
        """Score a frozen zero-shot model on the target held-out partition."""
        return self.dayk.score(selection, scorer, with_behavior=True)

    def score_adapted(
        self, selection: FrozenSelection, scorer: Callable[[CleanData], Any]
    ) -> Score:
        """Score a frozen adapted model on the same target held-out partition."""
        return self.dayk.score(selection, scorer, with_behavior=True)


def pair_experiments(day0: Experiment, dayk: Experiment) -> FixedPair:
    """Validate two sessions as a fixed, same-dataset Day-0 → Day-K pair."""
    if day0.source.dataset != dayk.source.dataset:
        raise ValueError("paired sessions must use the same dataset")
    day0_date = _session_date(day0.source.session_id)
    dayk_date = _session_date(dayk.source.session_id)
    if dayk_date <= day0_date:
        raise ValueError("Day-K session must be strictly later than Day-0")
    if (day0.view.mode, day0.view.bin_width_s) != (dayk.view.mode, dayk.view.bin_width_s):
        raise ValueError("paired sessions must use the same bin configuration")
    if day0.manifest.strategy != dayk.manifest.strategy:
        raise ValueError("paired sessions must use the same split strategy")
    if (day0.manifest.seed, day0.manifest.n_splits, len(day0.manifest.outer_folds)) != (
        dayk.manifest.seed, dayk.manifest.n_splits, len(dayk.manifest.outer_folds)
    ):
        raise ValueError("paired sessions must use compatible manifests")
    if day0.outer_fold.index != dayk.outer_fold.index:
        raise ValueError("paired experiments must use the same outer fold index")
    return FixedPair(day0, dayk)


def _session_date(session_id: str):
    match = re.search(r"(?:^|_)(\d{8})(?:_|$)", session_id)
    if match is None:
        raise ValueError(f"session ID does not contain a YYYYMMDD date: {session_id!r}")
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError as error:
        raise ValueError(f"session ID contains an invalid date: {session_id!r}") from error
