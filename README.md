# Noise Remover (React + Python)

## Backend (Python)

Install:

```bash
python -m pip install -r backend/requirements.txt
```

Run from the repository root:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

If you are inside the `backend/` directory instead, use:

```bash
uvicorn app:app --reload --port 8000
```

Health check: `http://127.0.0.1:8000/api/health`

## Frontend (React + TS + Tailwind)

Install:

```bash
cd frontend
npm install
```

Run:

```bash
cd frontend
npm run dev
```

Open the URL Vite prints (usually `http://127.0.0.1:5173`).

## Advanced settings (what they do)

- **Strength**: main denoise amount (0–1). Higher removes more noise, may add artifacts.
- **Residual gate**: removes the “last 5–15%” hiss after denoise. Too high can dull “S/SH” sounds.
- **HF boost (dB)**: extra suppression for high-frequency hiss.
- **Noise percentile**: how aggressively we estimate “noise-only” frames (lower = more aggressive).

