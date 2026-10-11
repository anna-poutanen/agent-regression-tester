# Contributing locally

Python 3.11 or later is required. Run `make setup`, then `make check`.

The package uses `src/agent_regression`, with pure artifact/check code separate from the framework adapter. Add behavior tests for meaningful changes, including a deliberately broken agent when changing interception. Tests use FunctionModel and block real provider requests in `tests/conftest.py`.

Before changing supported Pydantic AI versions, run the sentinel, retry, malformed-call, parallel-ordering, timeout, and concurrency tests against the new version. Preserve the fail-closed boundary: replay must never call the original function as a fallback.

Use `make demo` to inspect the successful workflow and `.venv/bin/agent-regression demo --broken` to inspect a seeded regression. The broken command must exit 1. Generated reports belong in `reports/` and are ignored by Git.

Scenarios are reviewed behavior contracts. Suggestions derived from recorded calls do not establish that those calls were correct. Whenever changing checks or fixtures, review the underlying policy again and reset `reviewed` until that review is complete.

Do not put API keys or customer records in fixtures. All committed examples are synthetic. `.env` files are ignored, but fixtures are ordinary JSON that may be committed.

The CI workflow is prepared for pull requests. Local development does not require GitHub, a remote repository, or provider credentials.
