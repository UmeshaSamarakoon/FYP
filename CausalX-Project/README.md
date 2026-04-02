# CausalX

CausalX is a deepfake-analysis prototype with:

- a FastAPI backend for upload, inference, and result retrieval
- a Vite/React frontend for the analysis UI
- checked-in Step46 model artifacts for the default local/runtime scoring path

## Repo Layout

- `backend/`: API, inference pipeline, scripts, model artifacts, notebooks
- `frontend/`: Vite frontend
- `backend/docs/`: demo and evaluation notes

## Run Locally

### 1. Start the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn src.cvi.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend local URL:

- `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/health`

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Frontend local URL:

- `http://127.0.0.1:5173`

The frontend defaults to the local backend at `http://127.0.0.1:8000`.

## Hosted URLs

- Frontend: `https://causalx-frontend.onrender.com`
- Backend: `https://causalx-backend.onrender.com`

The hosted services run on free Render instances, so cold starts and long video timeouts are expected.

## Notes

- The default scoring path expects the checked-in Step46 assets under `backend/models/`.
- Local uploads, caches, previews, and build outputs are generated at runtime and should not be committed.
- Notebook and experiment history is kept under `backend/notebooks/` and `backend/evidence/`.
