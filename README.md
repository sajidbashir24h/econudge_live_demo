# EcoNudge Live Prototype

EcoNudge is a sustainable recommender-system prototype for fashion e-commerce. It shows how a recommendation stack can balance user relevance, business performance, and sustainability impact inside one interactive decision-support dashboard.

This `live/` package is the deployable demo edition of the project. It is designed for public hosting on Streamlit Community Cloud or a similar lightweight platform, with fast startup, compact assets, and no dependency on the full local training pipeline.

## What This Demo Shows

- an interactive Streamlit dashboard with the same core user flow as the original project
- baseline vs sustainable recommendation comparison
- manager-controlled strategy weights for deadstock, margin, return risk, logistics, and loyalty
- live charts and score snapshots for coverage, popularity bias, and trust-related signals
- recommendation-level rationale and optional Gemini-powered explanation text
- in-app documentation for both technical and non-technical audiences

## Why This Repository Exists

The full EcoNudge research project includes data engineering, model training, evaluation, notebook workflows, and report assets. That full setup is valuable for research and reproducibility, but it is heavier than what we want for a public demo.

This live prototype keeps the experience and decision logic visible while replacing heavyweight runtime dependencies with compact precomputed assets and lightweight in-memory scoring. The goal is to make the project easy to review, easy to deploy, and strong as a portfolio artifact.

## What Is Included Here

- `app.py`: deployable Streamlit application
- `assets/demo_bundle.json`: compact precomputed artifact used by the live app
- `build_live_bundle.py`: helper script to rebuild the compact artifact from the main project outputs
- `requirements.txt`: minimal deployment dependencies
- `runtime.txt`: Python runtime pin for cloud deployment
- `.streamlit/config.toml`: Streamlit app configuration
- `.streamlit/secrets.example.toml`: example secrets file for Gemini setup

## What Is Intentionally Limited Here

This public demo does not include the full training and experimentation stack.

- no raw H&M dataset files
- no large model pickle files
- no notebook-driven full pipeline execution at app runtime
- no dependency on `src/` from the original project
- no Ollama or local Gemma serving requirement

Instead, recommendations are generated from a compact curated asset that preserves representative user, item, and strategy variation while staying lightweight enough for free cloud deployment.

## Current Live Bundle Scope

- 50 representative demo users
- expanded candidate pools for stronger scenario coverage
- broader item variation than the initial compact prototype
- lightweight artifact size suitable for public hosting

## Relationship To The Full Research Project

This repository is the demo surface of a larger EcoNudge system. The full project contains:

- complete data engineering
- baseline recommender training
- strategy-aware reranking experiments
- evaluation notebooks and reporting
- academic write-up and implementation details

If you publish this `live/` folder as a separate repository, add the URL of the full project here:

- Full research repository: `https://github.com/TuWienProjects/tuw-cdl-sustainable-recsys`

If you keep `live/` inside the main repository, you can replace that placeholder with the actual repository link before publishing.

## Project Background

EcoNudge was built around a real recommender-systems problem in fashion e-commerce: standard ranking pipelines often optimize short-term relevance while under-serving long-tail items, leaving deadstock unsold, ignoring return-risk patterns, and making sustainability trade-offs invisible.

The project addresses that gap with a control-room-style interface where users can inspect how strategic priorities change ranking behavior. Rather than replacing relevance, EcoNudge makes those trade-offs explicit and adjustable.

Core themes demonstrated in the app:

- relevance preservation
- sustainability-aware reranking
- business-aligned inventory decisions
- interpretable recommendation behavior
- human-readable explanation support

## Run Locally

Use `Python 3.12` for this app. The pinned dependencies are aligned with the project environment and may not install cleanly on newer Python versions such as `3.14`.

If you already have the parent project virtual environment:

```powershell
cd live
..\.venv\Scripts\python -m streamlit run app.py
```

If you want a dedicated environment just for the live demo:

```powershell
cd live
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

## Gemini API Setup

The existing `Show Gemma explanation` toggle in the UI can use Gemini as the live explanation backend.

1. Get an API key from Google AI Studio.
2. Create `live/.streamlit/secrets.toml`.
3. Copy the structure from `.streamlit/secrets.example.toml`.
4. Add your real key.

Example:

```toml
GEMINI_API_KEY = "your-real-key"
GEMINI_MODEL = "gemini-2.5-flash-lite"
```

Behavior in the app:

- if Gemini is configured, the toggle uses Gemini for the explanation panel
- if Gemini is unavailable or fails, the app falls back to deterministic local explanation logic
- if the toggle is off, no LLM explanation is shown

## Deploy

This package is designed to be deployed as its own small public repository.

For Streamlit Community Cloud:

1. Create a new GitHub repository using the contents of `live/`.
2. Push this folder's files to the root of that new repository.
3. Add `GEMINI_API_KEY` in Streamlit Cloud Secrets if you want live API explanations.
4. Set `app.py` as the entrypoint.

Recommended committed files:

- `app.py`
- `assets/demo_bundle.json`
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.example.toml`
- `README.md`
- `.gitignore`

## Rebuild The Compact Demo Asset

If you want to regenerate the live artifact from the full project outputs:

```powershell
cd live
..\.venv\Scripts\python build_live_bundle.py
```

That script is intended for maintainers who also have access to the complete project workspace and generated artifacts.

## Suggested Repository Positioning

This demo repo works best when presented as:

- a deployable public-facing prototype
- a lightweight product showcase of the research
- a companion surface to the full EcoNudge implementation

For portfolio readers, the key message is that this repository demonstrates the decision logic, UX, and deployable system behavior, while the full research repository contains the complete experimentation and pipeline context.
