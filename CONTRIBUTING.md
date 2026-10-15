# Contributing

The full guide is in [docs/about/contributing.md](docs/about/contributing.md): setting up, the repository map, the standards (tests and docs with every
change, runnable examples, adapter conventions) and how a release is cut.

In short: branch from `master`, keep `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q &&
uv run mkdocs build --strict` green, add a changelog entry in `docs/about/changelog.md`, and describe in the pull request what you verified and what you
could not. Security issues: see [SECURITY.md](SECURITY.md).
