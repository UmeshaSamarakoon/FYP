# Frontend

## Local Setup

From the `frontend/` directory:

```bash
cp .env.example .env.local
npm install
```

## Run Locally

```bash
npm run dev
```

Local frontend URL:

- `http://127.0.0.1:5173`

By default the frontend calls:

- `http://127.0.0.1:8000`

If the backend is running somewhere else, update `VITE_API_URL` in `.env.local`.

## Hosted URLs

- Frontend: `https://causalx-frontend.onrender.com`
- Backend API: `https://causalx-backend.onrender.com`

The hosted deployment runs on free Render instances, so analysis requests can be slow or time out after idle periods.

## Useful Commands

```bash
npm run dev
npm run build
```
