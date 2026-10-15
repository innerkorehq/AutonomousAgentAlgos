# Contributing

Contributions are welcome under the project's [MIT license](license.md).

## Set up

```bash
git clone https://github.com/innerkorehq/AutonomousAgentAlgos.git
cd AutonomousAgentAlgos
uv sync --extra docs        # dev tools, Postgres/Redis clients, and the docs toolchain
uv run pytest -q            # the full suite (starts throwaway Postgres/Redis if their binaries exist)
uv run ruff check . && uv run ruff format --check .
uv run pyright                 # library, examples and scripts must stay at 0 errors
uv run mkdocs build --strict
```

The Postgres and Redis tests launch temporary servers on random ports and never touch yours (needs `initdb`/`postgres` and `redis-server`
installed; set `AAA_TEST_PG_BIN` if they are off `PATH`). Without them those tests skip. A full run takes about two minutes. CI (`.github/workflows/ci.yml`) runs lint, pyright and the whole suite on Python 3.11, 3.12 and 3.13 against real Postgres and Redis services.

## Repository map

| Path | |
|---|---|
| `src/aaa/core` | outcomes, errors, ids |
| `src/aaa/events` | events, stores, buses, projections, guards |
| `src/aaa/data` | items, codecs, blobs, index |
| `src/aaa/llm` | clients, gateways, handler, fakes |
| `src/aaa/tools` | tools, handler, guard |
| `src/aaa/runtime` | engine, budget, recovery |
| `src/aaa/algos` | the algorithms |
| `examples/` | runnable examples **embedded in the docs and executed by the tests** |
| `tests/` | unit, contract, cluster and multi-process tests; `tests/helpers/` has worker processes |
| `scripts/` | `live_smoke.py`, `bench_growth.py` |
| `docs/` | this site |

## Standards

- **Tests with every change.** Behavior changes need a test; bug fixes need a test that fails without the fix.
- **Docs with every user-visible change.** If it changes what a doc example prints, the example test will fail: update both.
- **Examples are tests.** New doc examples go in `examples/`, are registered in `tests/unit/test_examples.py`, and are embedded with
  `--8<-- "examples/<name>.py"`.
- **Style.** `ruff` with line length 100. Match the surrounding code: comment density, naming, idiom.
- **Be honest in docs.** State what is verified and what is not. Do not document behavior you have not run.

## Adding things

=== "An algorithm"

    1. Write a projection for its state and the algorithm class ([guide](../guides/algorithms/custom.md)).
    2. Derive every event with `event.caused(...)`; no I/O in `on_event`.
    3. Test with `FakeLLMClient` driven by request *content*; assert on outcome and on the event list.
    4. Add a guide section, an example, and a row in "Choosing an algorithm".

=== "A store or bus backend"

    1. Implement the protocol ([Extending AAA](../guides/extending.md)).
    2. Add it to the `store` fixture in `tests/unit/test_store_contract.py` and the `cluster` fixture in `tests/conftest.py`; the contract suites must pass unmodified.
    3. Document its durability and failure semantics honestly.

=== "A gateway preset"

    1. Add a function in `aaa/llm/gateways.py` that only sets URL, auth and routing headers.
    2. Verify the URL and header names against the vendor's documentation and cite the page in the docstring.
    3. Add a mocked-transport test, and say in the docs whether you verified it live.

## Releasing (maintainers)

Releases go to PyPI through **Trusted Publishing** (OIDC): the `release` workflow proves its identity to PyPI, so no API token
is stored in the repository.

**One-time setup**

1. On PyPI, add a *pending publisher* (Account → Publishing): project `autonomous-agent-algos`, owner `innerkorehq`, repository
   `AutonomousAgentAlgos`, workflow `release.yml`, environment `pypi`. Repeat on TestPyPI with environment `testpypi` if you want dry runs.
2. In the GitHub repository, create the environments `pypi` and `testpypi` (Settings → Environments). Add a required reviewer to `pypi`
   so a publish needs a human click.

**Each release**

1. Set `version` in `pyproject.toml`, run `uv lock`, and give the changelog entry its date. Merge to `master`.
2. Optional dry run: Actions → *release* → *Run workflow* publishes the build to TestPyPI.
3. Create a GitHub Release tagged `vX.Y.Z` (same number as `pyproject.toml`; the workflow refuses a mismatch). The workflow builds,
   runs `twine check --strict`, installs the wheel in a clean environment and runs the quickstart, repeats the full test suite
   against Postgres and Redis, and then publishes.
4. **`aaa-server` releases separately.** Set its `version` in `packages/aaa-server/pyproject.toml`, raise its dependency floor on the library from the development version to the released one (the workflow refuses a `dev` floor), release the library first, then create a GitHub Release tagged `server-vX.Y.Z`. The `release-server` workflow publishes with its own trusted publisher (project `aaa-server`, workflow `release-server.yml`, environment `pypi-server`).
5. A published version can never be replaced; if one is bad, *yank* it on PyPI and release the next number.

## Pull requests

Describe the problem, the approach, and how you verified it. Keep changes focused. CI for docs runs on merge to `master`; run the suite locally before opening a PR.
