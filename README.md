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

## Deployment on Render.com

This project can be deployed as separate services on Render.com: one for the backend API and one for the frontend.

### Backend Deployment

1. **Create Backend Service:**
   - Go to [Render.com](https://render.com) and sign in
   - Click "New" → "Web Service"
   - Connect your repository
   - Set the **Root Directory** to `backend/` (important!)
   - **Environment**: Docker
   - **Name**: Choose a name (e.g., "noise-remover-backend")
   - **Region**: Choose the closest region
   - **Branch**: main (or your default branch)

2. **Environment Variables** (if needed):
   - No specific variables required for basic deployment

3. **Deploy:**
   - Click "Create Web Service"
   - The Dockerfile in `backend/` will be used automatically
   - Backend will be available at `https://your-backend-service.onrender.com`
   - Health check: `https://your-backend-service.onrender.com/api/health`

### Frontend Deployment

1. **Create Frontend Service:**
   - Click "New" → "Web Service" (or "Static Site" for simpler setup)
   - Connect your repository
   - Set the **Root Directory** to `frontend/`
   - **Environment**: Docker (if using Dockerfile) or Node for auto-build
   - **Name**: Choose a name (e.g., "noise-remover-frontend")
   - **Region**: Same as backend
   - **Branch**: main

2. **For Docker deployment:**
   - Uses the Dockerfile in `frontend/`
   - Exposes port 80

3. **For Node.js auto-build (simpler):**
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`
   - No Dockerfile needed

4. **Environment Variables:**
   - Add `REACT_APP_API_URL=https://your-backend-service.onrender.com` to connect frontend to backend

5. **Deploy:**
   - Frontend will be available at `https://your-frontend-service.onrender.com`

### Configuration Notes

- **CORS**: The backend allows all origins (`*`) for development. For production, restrict to your frontend URL.
- **API Calls**: Update frontend to use the backend service URL for API requests.
- **Free Tier**: Both services can use Render's free tier, but monitor usage limits.
- **Custom Domain**: You can add custom domains to both services separately.
