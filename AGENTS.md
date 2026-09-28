# Repository Guidelines

## Project Structure & Module Organization

OpenAlfred is a Python 3.13 backend built with FastAPI and LangGraph. Application code lives in `src/`: routes in `routers/`, orchestration in `logic/`, integrations in `services/`, agent-callable functions in `tools/`, persistence in `db/` and `core/`, and retrieval in `rag/`. Windows voice and screen components live under `src/body/windows_system/`. Put automated tests in `tests/`; `probe_*.py` scripts are manual integration probes. Static audio, models, and screenshots belong in `assets/`. Do not commit runtime data from `data/`, `memory/`, `chroma_db/`, databases, or logs.

## Build, Test, and Development Commands

- `uv sync` installs the locked dependencies into `.venv/`. Use `uv`, never `pip`.
- `uv run langgraph dev` starts the LangGraph development server on port 2024.
- `cd src; uv run python -m app` starts the FastAPI service on port 7788.
- `./start-all.ps1` launches the local Windows service stack.
- `uv run python -m unittest discover -s tests -p "test_*.py"` runs the automated test suite.
- From `src/`, `uv run python -m rag.cli demo` exercises the RAG pipeline.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8: `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Type-hint new public functions and keep database, network, and ingestion paths asynchronous. Use `utils.logger.get_logger()` instead of `print()` in services. Keep routers thin by placing business logic in `services/`, `db/`, or `rag/`. No formatter or linter is configured; match nearby code and group standard-library, third-party, and local imports.

## Testing Guidelines

Tests use `unittest`, including `IsolatedAsyncioTestCase` and `unittest.mock`. Name files `test_<feature>.py`, classes `Test<Behavior>`, and methods `test_<expected_result>`. Cover success paths and async failures. Keep network-dependent experiments in `probe_*.py`; automated tests should mock external APIs and credentials. Run the full suite before submitting.

## Commit & Pull Request Guidelines

History primarily uses `feat:`, `fix:`, and `chore:` prefixes. Write imperative summaries (for example, `fix: handle IMAP ID after login`) and separate unrelated changes. Pull requests should explain behavior, note configuration or schema impacts, link issues, and list verification commands. Include screenshots for user-visible changes.

## Security & Configuration

Copy settings from `.env.example`; never commit `.env`, API keys, email credentials, JWT secrets, generated databases, or personal memory files. Preserve `user_id` isolation in every CRUD and RAG path. Document new environment variables in `.env.example` with safe defaults.
