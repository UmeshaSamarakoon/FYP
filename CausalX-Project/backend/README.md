# Backend

## What This Service Does

The backend provides:

- `POST /analyze` for synchronous video analysis
- `POST /analyze/async` and `GET /analyze/status/{job_id}` for polling-based analysis
- `GET /results/{analysis_id}` for stored results
- `GET /health` for readiness checks

The default inference path uses the tracked single-checkpoint CFN backbone in `backend/models/cfn_emb.pth` plus `backend/models/cfn_scaler.pkl`. Optional video-level Step46 scorers remain available, but they are opt-in.

## Local Setup

From the `backend/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run Locally

```bash
python -m uvicorn src.cvi.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Local endpoints:

- API root: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

## Environment Configuration

The backend reads runtime settings from `.env`.

Start from:

```bash
cp .env.example .env
```

Key settings:

- `CFN_EMB_MODEL_PATH`
- `CFN_SCALER_PATH`
- `CFN_RATIO_THRESH`
- `CFN_CAUSAL_THRESH`
- `CFN_REQUIRE_FLAG`
- `CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST`
- `CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR`

In the current checked-in configuration, the live frame pipeline is the default for both local runs and deployment. The checked-in multiseed Step46 manifest is not the default runtime path.

## Hosted Backend

- `https://causalx-backend.onrender.com`

Because the hosted service is on a free Render instance, cold starts and long analysis timeouts can happen.

If you deploy with the repo-root [render.yaml](/Users/venturit/Documents/GitHub/FYP/CausalX-Project/render.yaml), Render is pinned to the same live-pipeline env as local. For an already-created manual Render service, remove any stale `CFN_ENSEMBLE_MANIFEST_PATH` dashboard variable and mirror the same backend env values there instead.

## Project Files Worth Keeping

- `src/`: production API and inference code
- `models/fakeavceleb_best_step46_multiseed_manifest.json`
- `models/step46_fakeav_robust_constrained_*`
- `scripts/run_fakeav_mrdf_5fold_cv.py`
- `scripts/run_step46_fakeav_robust_constrained.sh`

## What Is Not Part Of The Core Runtime

These are useful for research, validation, or thesis support, but not required to boot the API:

- `notebooks/`
- `evidence/`
- most generated `outputs/`, `uploads/`, and local caches
