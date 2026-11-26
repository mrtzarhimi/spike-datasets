# Spike Datasets Implementation Plan

> **For agentic workers** — Use Superpowers subagent-driven development for implementation and verification before handing off each branch. Checkboxes track work that will happen during implementation.

**Goal** — Replace repeated Chewie and Jango data preparation with one installable package for within-session benchmarks and fixed Day-0 → Day-K comparisons.

**Architecture** — An XDS adapter indexes metadata and reads selected intervals. Common code defines binning and split manifests. Development and final scoring use the same manifests through separate entry points.

**Tech stack** — Python 3.10+, a `pyproject.toml` build, NumPy, h5py, pytest, and GitHub Actions. Python 3.10 is a proposed compatibility floor.

**Spec** — [Shared Chewie and Jango dataset protocol](../specs/2026-11-27-spike-datasets-design.md)

## Global constraints

- Use 20 ms complete bins, zero smoothing and zero neural lag by default.
- Use three grouped outer folds, split seed 42, and two grouped inner folds. Chronological mode uses the earliest groups for fit and latest groups for validation or held-out scoring.
- Noise defaults to 0.02 times per-unit population standard deviation, accepts factors from 0.0 to 0.05, and bounds standard normal draws at ±3. Every scale is fit on clean bins from that role's fit partition.
- Keep held-out neural and behavior values out of development. Train inner transformations on inner fit only. Freeze selected settings, refit on outer held-in data, and open the held-out partition for scoring. Complete selection for every compared arm and fold before revealing final scores.
- Reuse a manifest across methods and comparison arms. Use native task targets, explicit experiment labels and immutable input identities.
- Use synthetic fixtures in public CI. Keep real data, outputs and machine-specific config outside Git and distributions.

## Agent queue and branch ownership

| Task | Title | Depends on | Branch |
| --- | --- | --- | --- |
| SD-01 | Bootstrap the Python package and shared types | — | `chore/sd-01-package` |
| SD-02 | Read Chewie and Jango XDS metadata and slices | SD-01 | `feat/sd-02-xds` |
| SD-03 | Build split-safe trial and fixed-time views | SD-01 | `feat/sd-03-binning` |
| SD-04 | Persist three-fold and chronological split plans | SD-03 | `feat/sd-04-splits` |
| SD-05 | Fit and apply small training-only noise | SD-01 | `feat/sd-05-noise` |
| SD-06 | Enforce the development and final-score boundary | SD-02, SD-03, SD-04, SD-05 | `feat/sd-06-evaluation` |
| SD-07 | Support fixed cross-day pairs and document both workflows | SD-06 | `feat/sd-07-cross-day` |
| SD-08 | Verify a public-ready package and smoke-test both datasets | SD-07 | `chore/sd-08-release` |

SD-02, SD-03, and SD-05 can proceed in parallel after SD-01 merges. Start SD-04 when SD-03 merges. Start each task from the latest base containing its dependencies and restrict edits to its owned files. The implementation-model request is **GPT-6 Luna**. Check that the dispatch environment offers it before assigning coding work.

## Review focus

- MATLAB v7.3 stores channel-major signal arrays and referenced text, with channel counts that can differ by day. SD-02 tests axes, references and variable channel counts.
- Touching intervals, overlaps, incomplete tails, timebase gaps and NaN behavior require explicit treatment. SD-03 tests each case with hand-computed coverage.
- Unequal group sizes can create empty or highly imbalanced folds. SD-04 tests a dominant group and too few groups.
- A constant training unit can have large variance on test. SD-05 requires zero noise scale for that unit, and SD-06 poisons all held-out signals to prove fit and tuning invariance.
- Adaptation reads can expose target behavior indirectly through a shared loader. SD-07 tests the streams accessed by zero-shot and adaptation workflows independently.

## SD-01 · Bootstrap the Python package and shared types

**Owns** `pyproject.toml`, `.gitignore`, `src/spike_datasets/__init__.py`, `src/spike_datasets/types.py`, `tests/test_types.py`.

**Produces** `Source(path: Path, dataset: Dataset, session_id: str, revision: str)`, with `Dataset = Literal["chewie", "jango"]`. `SessionMeta` carries the dataset and session ID, revision, native bin width in seconds, native timestamps, ordered unit IDs, target field and recorded units, and ordered `TrialMeta` records. Each `TrialMeta` carries the original row ID, start and end in seconds, and optional cue, outcome and direction. `RawSpan(start: int, stop: int)` denotes native indices on a half-open grid. `BinnedBlock` carries sample-by-unit counts, optional sample-by-axis targets and a matching finite-target mask, and a raw span and timestamp for each bin. Partition and group identifiers refer to original metadata IDs.

- [ ] Write `test_raw_span_rejects_reversed_and_empty_bounds` and `test_binned_block_rejects_mismatched_axes`. Run `python -m pytest tests/test_types.py` and verify the missing-package or contract failure.
- [ ] Add the build metadata with NumPy and h5py as the runtime dependencies. Export the shared types and validation contract. Use explicit identifiers and preserve count and time dtypes.
- [ ] Run `python -m pip install -e '.[test]'` and rerun `python -m pytest tests/test_types.py`. Verify the import resolves through the installed package.
- [ ] Commit as `chore: bootstrap spike datasets package`.

## SD-02 · Read Chewie and Jango XDS metadata and slices

**Owns** `src/spike_datasets/xds.py`, `tests/fixtures/xds.py`, `tests/test_xds.py`.

**Consumes** `Source`, `SessionMeta`, and `RawSpan` from SD-01.

**Produces** `discover_sessions(root: Path, dataset: Dataset, *, revision: str) -> tuple[Source, ...]`, `inspect_session(source: Source) -> SessionMeta`, and `read_span(source: Source, span: RawSpan, *, with_behavior: bool) -> RawBlock`. Source IDs use the recording file stem. Discovery lists the extracted dataset's matching session files in date and recording order. `RawBlock` carries the requested native count rows in time-major order and optional time-major target rows.

- [ ] Build tiny synthetic MATLAB v7.3-style XDS fixtures with asymmetric matrix shapes, referenced unit names, two-dimensional targets and 1 ms timestamps. Include both Jango `force` and `curs_v` with deliberately different values.
- [ ] Write `test_reads_channel_major_counts_and_explicit_target`. Assert source A reads `curs_v`, source B reads `force`, a request for `[2, 5)` returns exactly three rows, and unit names retain file order when sessions have different unit counts. Add failures for missing required fields, invalid references, and mismatched time axes. Instrument file access to assert metadata inspection slices no spike or target values and `with_behavior=False` slices no behavior. Run `python -m pytest tests/test_xds.py` and verify failure.
- [ ] Implement filename-based discovery and one reader with thin dataset specifications. Test deterministic discovery with both datasets and an unrelated file present. Decode MATLAB character and reference metadata, derive the time axis from schema and timestamps, and read only requested spans with h5py slicing. Report missing required fields by source and field. Open and close files within reader calls.
- [ ] Rerun `python -m pytest tests/test_xds.py`. Commit as `feat: read Chewie and Jango XDS slices`.

## SD-03 · Build split-safe trial and fixed-time views

**Owns** `src/spike_datasets/binning.py`, `tests/test_binning.py`.

**Consumes** SD-01 shared types. Tests can use arrays and a fake span reader while SD-02 is in progress.

**Produces** `plan_view(meta: SessionMeta, *, mode: Literal["trial", "fixed"], bin_width_s: float = 0.02) -> ViewPlan` and `materialize(view: ViewPlan, group_ids: Collection[int], read_span: SpanReader, *, with_behavior: bool) -> ViewData`. `ViewPlan` describes ordered groups, trial IDs, owned native intervals and planned bin spans. `ViewData` carries binned blocks with group IDs and trial IDs where applicable. `SpanReader` accepts a `RawSpan` and keyword `with_behavior` and returns a native count matrix plus optional target matrix.

- [ ] Write a hand-computed case on a 1 ms grid with 2 ms bins. Assert a spike series `[1, 2, 0, 3, 5]` becomes `[3, 3]` for a five-sample trial, the fifth sample is excluded, and behavior `[2, 4, 6, 8, 10]` becomes `[3, 7]`. Test a per-axis NaN mask, positive and invalid widths, a nonuniform timebase, intertrial gaps, incomplete tails, and transitive overlaps. With 4 ms bins, assert a short trial `[3, 6)` groups `[0, 4)` and `[5, 9)` together, while a standalone short trial becomes gap time in the fixed view. Assert touching half-open intervals can split, fixed bins crossing group ownership are omitted, and both views retain the same group IDs. A spy reader must observe requests only for the selected groups.
- [ ] Run `python -m pytest tests/test_binning.py` and verify failure.
- [ ] Implement metadata-only planning and materialization. Convert time bounds to indices with left insertion. Form groups using all valid bounds, attach components without a complete trial bin as gap time, and apply the same ownership in both views. Report excluded trials and dropped spans and reject invalid spike values when reading a selected role.
- [ ] Rerun `python -m pytest tests/test_binning.py`. Commit as `feat: add trial and fixed-time binning`.

## SD-04 · Persist three-fold and chronological split plans

**Owns** `src/spike_datasets/splits.py`, `tests/test_splits.py`.

**Consumes** `ViewPlan` with stable ordered group IDs, session identity, and raw-span ownership.

**Produces** `plan_splits(view: ViewPlan, *, strategy: Literal["kfold", "chrono"] = "kfold", n_splits: int = 3, seed: int = 42) -> SplitManifest`. A manifest contains `OuterFold` records with fit and held-out group IDs and nested `InnerFold` fit and validation IDs. `SplitManifest.to_json() -> str` emits versioned canonical JSON. `load_manifest(text: str, view: ViewPlan) -> SplitManifest` validates identity, coverage, and membership before reuse.

- [ ] Write `test_every_group_is_held_out_once_and_inner_is_nested`. Assert deterministic IDs for the same seed, zero raw overlap, no empty roles and the exact union of K-fold test groups. For chronological groups `[0, 1, 2, 3, 4, 5]`, expect held-out `[4, 5]`, inner fit `[0, 1]`, and inner validation `[2, 3]`. Include imbalanced groups, insufficient groups, a role with groups but zero bins in the selected view, duplicate or out-of-range group IDs, a mismatched view manifest, and a JSON round trip.
- [ ] Run `python -m pytest tests/test_splits.py` and verify failure.
- [ ] Permute stable group IDs with a seeded NumPy generator and use that order to break ties as groups are packed by descending eligible trial count into the currently lightest fold. Initialize each fold with one group. Use identical group weights across views and preserve timestamp order when materializing each role. Reject empty roles and ambiguous chronology.
- [ ] Rerun `python -m pytest tests/test_splits.py`. Commit as `feat: add grouped and chronological split manifests`.

## SD-05 · Fit and apply small training-only noise

**Owns** `src/spike_datasets/noise.py`, `tests/test_noise.py`.

**Produces** internal `fit_noise(clean_fit: Iterable[NDArray], *, factor: float = 0.02) -> NoiseState` and `augment_fit(counts: NDArray, state: NoiseState, *, rng: numpy.random.Generator) -> NDArray`. SD-06 binds each fitted state to a session, fold and view.

- [ ] Write `test_fit_noise_uses_per_unit_population_std_and_bounds_deltas`. For `[[0, 4], [2, 4]]`, assert scales before multiplying by the factor are `[1, 0]`. Assert deterministic draws for a repeated seed, a maximum default change of 0.06 on the first unit, exactly zero change on the second unit, and unchanged input. Add factor-zero, empty fit, inconsistent channel shape, and nonfinite or out-of-range factor cases.
- [ ] Run `python -m pytest tests/test_noise.py` and verify failure.
- [ ] Implement the bounded draw and stable population variance in floating point over the supplied clean fit bins. Expose clean integer counts for count likelihoods alongside float training features.
- [ ] Rerun `python -m pytest tests/test_noise.py`. Commit as `feat: add bounded fit-only spike augmentation`.

## SD-06 · Enforce the development and final-score boundary

**Owns** `src/spike_datasets/evaluation.py`, curated public exports in `src/spike_datasets/__init__.py`, `tests/test_evaluation.py`.

**Consumes** SD-02 reader, SD-03 view plan, SD-04 manifest, and SD-05 noise functions.

**Produces** `open_experiment(source: Source, view: ViewPlan, manifest: SplitManifest, outer_fold: int) -> Experiment`. The experiment exposes `inner_data(index, *, noise_factor=0.02, noise_seed, with_behavior=True) -> FitAndValidation`, `lock_selection(settings: Mapping[str, JsonValue]) -> FrozenSelection`, `refit_data(selection, *, noise_seed, with_behavior=True) -> FitData`, and `score(selection, scorer, *, with_behavior=True) -> Score`. The locked settings include the selected noise factor. `FitData` exposes both clean counts and optional augmented features. Selection is bound to source, manifest and fold. The scorer receives only the clean final partition and uses a caller-supplied frozen model.

- [ ] Write an instrumented nested experiment that fails on any attempt to read an outer held-out spike or target row before final scoring. Verify final scoring opens only the declared held-out rows. Poison held-out spikes and targets and assert identical manifest membership, inner-fit noise state, inner validation, chosen settings, and outer refit inputs with the same seeds. Poison inner validation and assert inner-fit noise state stays identical. Assert repeated fit materializations use the same noise and clean scoring inputs remain clean. Reject selections from another fold or source.
- [ ] Run `python -m pytest tests/test_evaluation.py` and verify failure.
- [ ] Implement development, lock, refit, and score entry points. Fit noise inside each fit call and materialize one seeded draw in canonical bin order. Return canonical settings and identity alongside scores. Leave model fitting and score persistence to the caller. Validate that role assignments keep overlapping native samples together.
- [ ] Rerun `python -m pytest tests/test_evaluation.py` across both view modes and strategies. Commit as `feat: isolate development from held-out scoring`.

## SD-07 · Support fixed cross-day pairs and document both workflows

**Owns** `src/spike_datasets/pairs.py`, `README.md`, `examples/within_session.py`, `examples/cross_day.py`, `tests/test_pairs.py`.

**Consumes** SD-06 experiment interfaces for source and target.

**Produces** `pair_experiments(day0: Experiment, dayk: Experiment) -> FixedPair`. A pair validates dataset, session order, bin configuration, split strategy and fold index. Its supported adaptation methods expose target fit and inner validation with behavior disabled. Scoring uses the target held-out partition only, after both choices are locked. A zero-shot pair uses no target signal before scoring.

- [ ] Write `test_zero_shot_does_not_read_target_before_score` and `test_adaptation_reads_only_target_held_in_spikes`. Use stub fit and scoring functions and a spy reader. Assert that target force or velocity is first read during scoring, source has no held-out reads, method arms share exact target bins, and the pair rejects a reversed date or mismatched dataset, configuration or fold index.
- [ ] Run `python -m pytest tests/test_pairs.py` and verify failure.
- [ ] Implement the two fixed-pair flows and runnable examples with explicit settings, seeds, split manifests, target units and local file arguments. Document within-session and cross-day comparisons, paired reporting, the clean baseline, and why noisy float inputs and integer count observations are distinct.
- [ ] Run `python -m pytest tests/test_pairs.py` and both examples against synthetic fixtures. Commit as `feat: support fixed cross-day benchmark pairs`.

## SD-08 · Verify a public-ready package and smoke-test both datasets

**Owns** `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `tests/test_release_smoke.py`, and the maintainer-approved `LICENSE`. Coordinate packaging metadata changes with SD-01.

- [ ] Add CI for the supported Python versions with synthetic regression tests and no dataset downloads. Build the sdist and wheel, install the wheel into an isolated environment, import outside the checkout, and verify that the distributions contain only intended package and project files.
- [ ] Write and run an opt-in smoke test with each real dataset and both views. Record source/session identifiers, timebase, target keys, eligible groups and fold counts. Confirm deterministic fold IDs and zero cross-role raw overlap. Exercise one cross-day pair using local inputs where available. Treat skipped real-data cases as unverified.
- [ ] Record the dataset DOI, local acquisition steps and release commands. Use only authorized license text and check code provenance before release. Verify `python -m pytest`, `python -m build`, and `python -m twine check dist/*`. Commit as `ci: verify public package and data protocol`.

## Git workflow

1. Create the package repository after the GitHub owner and repository name are confirmed. Use `main` as the integration branch. Once the GitHub destination exists, create one issue per SD task and link its dependencies. Keep local workspace details out of public issue bodies.
2. Use one branch and worktree per agent task. Assign disjoint file ownership in the first parallel wave. Keep each PR tied to one issue, include the tests run, and note any change to the default protocol or manifest schema.
3. Require passing synthetic CI and review on `main`. Squash merge in dependency order. When evaluation and cross-day PRs are reviewed, inspect the actual reader calls and the held-out poisoning tests.
4. Before the first public tag, verify both built distributions, code licensing, data citations, and local smoke results. Tag the reviewed commit `v0.1.0` after release approval. Use the same protected `main` and short-lived PR flow for fixes.
