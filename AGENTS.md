# AGENTS.md

Guidelines and context for AI coding agents (GitHub Copilot, etc.) working in this repository.

---

## Project overview

**lightcurve-demo** is a single-file Streamlit web application that demonstrates how astronomical lightcurves are created. A webcam captures a light source in real time; the app measures pixel flux inside a circular aperture and plots it as a normalised lightcurve. It is designed as a public interactive exhibit.

The entire application lives in **`lc_demo.py`**.

---

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python ≥ 3.12 |
| Package management | [uv](https://docs.astral.sh/uv/) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Charting | [Plotly](https://plotly.com/python/) (`plotly.graph_objects`) |
| Webcam / image processing | [OpenCV](https://opencv.org/) (`opencv-python`) |
| Numerics | [NumPy](https://numpy.org/) |
| Linting / formatting | [Ruff](https://docs.astral.sh/ruff/) |
| CI | GitHub Actions (`.github/workflows/ci.yml`) |

---

## Running the app

```bash
# Install uv (once)
curl -Ls https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run
uv run streamlit run lc_demo.py
```

The app opens at http://localhost:8501.

---

## Running tests

```bash
uv run pytest -v                   # run all tests with coverage report
uv run pytest -v --no-cov          # skip coverage (faster)
uv run pytest tests/test_lc_demo.py::TestNormalise  # run one class
```

Coverage is reported to the terminal and written to `htmlcov/` (open `htmlcov/index.html` in a browser).

The testable surface is the pure-Python business logic: `_measure_flux`, `_normalise`, `_build_figure`, and `_get_frame`. Streamlit session-state functions (`_init_state`, `_reset_lc`, `_open_camera`, `_release_camera`) and `main()` require a live Streamlit context and are covered by manual/integration testing only.

---

## Linting and formatting

```bash
uv run ruff check .       # lint
uv run ruff format .      # auto-format
uv run ruff format --check .   # format check (CI mode)
```

Ruff configuration is in `pyproject.toml` under `[tool.ruff]`.  
Line length is 100. Target version is `py312`.

---

## Code conventions

- **Python 3.12+ features are expected and preferred:**
  - Use `type Foo = ...` (PEP 695) for type aliases — not `TypeAlias`
  - Use `X | Y` union syntax — not `Union[X, Y]` or `Optional[X]`
  - Do **not** add `from __future__ import annotations` — it is not needed on 3.12+
  - Use `list[T]`, `dict[K, V]`, `tuple[T, ...]` built-in generics directly
- All functions should have return-type annotations.
- Module, class, and public function docstrings are required.
- Private helpers are prefixed with `_`.

## Streamlit patterns

- All mutable UI state lives in `st.session_state` — never in module-level globals.
- `_init_state()` sets default values for every key on first run; call it at the top of `main()`.
- Live updates use `time.sleep(TICK_S)` followed by `st.rerun()` — do not use `st.experimental_rerun`.
- Use `st.empty()` placeholders for in-place updates to avoid re-rendering the full page.
- Use `use_container_width=True` on all `st.image()`, `st.plotly_chart()` calls for responsiveness.
- Page config (`st.set_page_config`) must be the **first** Streamlit call in `main()`.

## Camera conventions

- The `cv2.VideoCapture` object is stored in `st.session_state.cap` and opened once; reuse it across reruns.
- Release the camera with `_release_camera()` when the device index changes or on app shutdown.
- Frames are read as BGR (OpenCV default) and converted to RGB immediately for all downstream use.
- The circular aperture mask is built with `cv2.circle` + `cv2.bitwise_and`; do not change this logic without understanding the photometry implications.

## Photometry conventions

- `_measure_flux()` returns a `FluxTriple` (r, g, b); for greyscale all three are equal.
- `_normalise()` scales flux relative to the baseline reading at tick 0, clamped to 100.
- Guard against division by zero: if `normal_factor[i] == 0`, return `0.0` (handled in `safe_div`).

---

## What NOT to do

- Do not add conda, pip-only `requirements.txt`, or any setup shell scripts — uv + `pyproject.toml` is the single source of truth.
- Do not hardcode paths like `/home/panoptes/Pictures/` — use `pathlib.Path.home()` or Streamlit download buttons.
- Do not use `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Union`, or `typing.Tuple` — use the built-in generic forms.
- Do not import `from __future__ import annotations`.
- Do not use `app.exec_()` or any PyQt/PySide code — the GUI layer is entirely Streamlit.
- Do not add `seaborn` — it was listed in the old README but is not used anywhere.

---

## File map

```
lightcurve-demo/
├── lc_demo.py              # Main Streamlit application (entire app)
├── pyproject.toml          # Project metadata, dependencies, ruff config
├── uv.lock                 # Locked dependency tree (commit this)
├── .streamlit/
│   └── config.toml         # Streamlit server/theme defaults (dark, wide)
├── .github/
│   └── workflows/
│       └── ci.yml          # Ruff lint + format check on push/PR
├── .gitignore
├── README.md
├── AGENTS.md               # This file
├── LICENSE
└── screenshot.png
```
