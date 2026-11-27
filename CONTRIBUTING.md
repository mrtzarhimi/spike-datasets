# Contributing and release checks

## Dataset citation and local data

The real recordings are not included in this repository, downloaded by CI, or
bundled in package archives. Acquire them from the [Dryad record for “Using
adversarial networks to extend brain computer interface decoding accuracy over
time”](https://doi.org/10.5061/dryad.cvdncjt7n). Cite the dataset as:

> Ma, Xuan; Rizzoglio, Fabio; Thacker, Stephanie; Ethier, Christian; Perich,
> Matthew; Miller, Lee (2023). Data from: Using adversarial networks to extend
> brain computer interface decoding accuracy over time [Dataset]. Dryad.
> https://doi.org/10.5061/dryad.cvdncjt7n

Download the current Dryad release and extract `Chewie_CO_2016.7z` and
`Jango_ISO_2015.7z` locally. The MATLAB files should be under separate
directories, for example:

```text
data/real/chewie/Chewie_CO_2016/*.mat
data/real/jango/Jango_ISO_2015/*.mat
```

The repository ignores `data/` and MATLAB files. Keep archives, extracted
recordings, checksums, and experiment outputs out of Git. Point the optional
smoke tests at the directories containing the `.mat` files; no data download is
performed by these commands:

```bash
SPIKE_DATASETS_CHEWIE_ROOT=data/real/chewie/Chewie_CO_2016 \
SPIKE_DATASETS_JANGO_ROOT=data/real/jango/Jango_ISO_2015 \
.venv/bin/python -m pytest -s -q tests/test_release_smoke.py
```

That command inspects every discovered session in both trial and fixed-time
views, replays the seeded folds, checks that role raw spans do not overlap, and
prints each session ID, native bin width, target field, eligible groups, and
outer/inner fold group counts. It also exercises a fixed-time Jango Day-0 to
Day-K zero-shot score. If either root is unset, its real-data cases are
reported as skipped; skipped cases are not evidence of a real-data check.
The observed local results are recorded in
[`docs/superpowers/releases/2026-11-27-sd-08-real-data-smoke.md`](docs/superpowers/releases/2026-11-27-sd-08-real-data-smoke.md).

## Local development and synthetic smoke tests

Create or activate a Python 3.10+ environment, then install the project and its
test dependency:

```bash
python -m pip install -e '.[test]'
python -m pytest
PYTHONPATH=src python examples/within_session.py --synthetic
PYTHONPATH=src python examples/cross_day.py --synthetic
```

The examples create temporary synthetic XDS recordings and never need the real
datasets. The public GitHub Actions workflow runs the full synthetic test suite
on Python 3.10 through 3.14, and builds/checks the distributions on Python 3.12.
Hosted CI status is available from the Actions run for the commit;
a local run does not establish hosted CI status.

## Release checks

Run from the repository root with the release environment active:

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

Inspect the wheel and source archive contents before publishing. Confirm that
they contain only project metadata, documentation, and intended package code;
they must not contain local configuration, recordings, archives, caches, or
experiment outputs. Install the wheel into a fresh environment and import it
while the working directory is outside the checkout to ensure the import comes
from the installed wheel.

Before publication, the maintainer must also approve a code license and confirm
that any reused code has compatible provenance and required upstream notices.
No license has been selected or included by this repository. That decision is a
publication blocker; do not add a license file or publish until the maintainer
supplies or approves the license text.
