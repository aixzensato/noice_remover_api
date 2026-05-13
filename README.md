---
title: Noise Remover API
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Noise Remover API

## Backend (Python)

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Health check: `http://127.0.0.1:8000/api/health`


## Advanced settings (what they do)

- **Strength**: main denoise amount (0–1). Higher removes more noise, may add artifacts.
- **Residual gate**: removes the "last 5–15%" hiss after denoise. Too high can dull "S/SH" sounds.
- **HF boost (dB)**: extra suppression for high-frequency hiss.
- **Noise percentile**: how aggressively we estimate "noise-only" frames (lower = more aggressive).

## Deployment on Hugging Face Spaces (Recommended)

The backend is deployed on Hugging Face Spaces for better performance (2 vCPU, 16GB RAM — free tier).

### Backend Deployment

1. Go to [huggingface.co](https://huggingface.co) and create a free account
2. Click **New Space**
3. Fill in:
   - **Space name**: `noise-remover-api` (or any name you like)
   - **SDK**: Docker ← important
   - **Visibility**: Public
4. Go to the **Files** tab and upload these files from the `backend/` folder:
   - `app.py`
   - `denoise.py`
   - `index.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
5. The Space will build automatically. Once running, your API will be live at:
   ```
   https://YOUR-HF-USERNAME-noise-remover-api.hf.space
   ```
6. Health check: `https://YOUR-HF-USERNAME-noise-remover-api.hf.space/api/health`

> **Note**: Free Spaces may pause after 48 hours of inactivity. They wake up automatically on the next request (cold start takes ~30 seconds).

## Deployment on Render.com (Alternative — not recommended for large files)

> ⚠️ Render free tier only provides 0.1 CPU and 512MB RAM. Audio files above ~10MB may fail or timeout. Use Hugging Face Spaces for the backend instead.

### Backend Deployment

1. **Create Backend Service:**
   - Go to [Render.com](https://render.com) and sign in
   - Click "New" → "Web Service"
   - Connect your repository
   - Set the **Root Directory** to `backend/`
   - **Environment**: Docker
   - **Name**: Choose a name (e.g., `noise-remover-backend`)
   - **Region**: Choose the closest region
   - **Branch**: main (or your default branch)

2. **Deploy:**
   - Click "Create Web Service"
   - The Dockerfile in `backend/` will be used automatically
   - Backend will be available at `https://your-backend-service.onrender.com`
   - Health check: `https://your-backend-service.onrender.com/api/health`

## Configuration Notes

- **CORS**: The backend allows all origins (`*`) for development. For production, restrict to your allowed origins.
- **API Calls**: Use the backend service URL for API requests.
- **File size limit**: Supports up to 50MB audio files when hosted on Hugging Face Spaces.
- **Custom Domain**: You can add a custom domain to the backend service.
