# Repository Guidelines

## Project Structure & Module Organization
Core application code lives at the repo root and in the top-level packages. `app.py` boots the FastAPI app and mounts routes from `api/routes/`. Business logic is in `services/`, ML model and embedding helpers are in `ml/`, shared helpers are in `utils/`, and environment-driven settings are in `core/config.py`. Tests live in `tests/`. Cached model files and FAISS artifacts are stored in `models/` and should be treated as generated assets, not hand-edited source.

## Build, Test, and Development Commands
Use Python 3.11 and a virtual environment.

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
pytest
```

`uvicorn app:app --reload` starts the API locally at `http://127.0.0.1:8000`. `pytest` runs the current test suite in `tests/test_services.py`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module-level docstrings where useful, `snake_case` for functions and variables, `PascalCase` for classes, and concise endpoint/service names such as `read_movie_history` or `build_user_profile`. Keep route handlers thin and push recommendation logic into `services/`. No formatter or linter is configured today, so match the surrounding file style and keep imports grouped logically.

## Testing Guidelines
This project uses `pytest`. Add tests next to the existing suite in `tests/test_services.py` or split into additional `tests/test_*.py` files as coverage grows. Name tests `test_<behavior>()` and prefer focused unit tests for utilities, scoring, profile generation, and recommendation filtering. When changing API behavior, add at least one regression test for the affected path.

## Commit & Pull Request Guidelines
The visible local history is minimal, so there is no strong subject-line convention to copy. Use short, imperative commit messages such as `Add fallback for empty catalog` or `Refactor movie client setup`. PRs should include a brief summary, note any config or API contract changes, list test coverage (`pytest`), and include sample requests/responses when endpoint behavior changes.

## Security & Configuration Tips
Configuration is loaded from `.env`. Required values include `MOVIE_API`; model and cache paths also come from env vars. Do not commit secrets or environment-specific tokens. Treat `models/*.index` and `models/*.npz` as rebuildable cache outputs.
