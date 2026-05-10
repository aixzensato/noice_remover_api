from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from denoise import DenoiseConfig, denoise_audio


app = FastAPI(title="Noise Remover API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/denoise")
async def denoise_endpoint(
    file: UploadFile = File(...),
    strength: float = Form(0.85),
    residual_gate: float = Form(0.35),
    hf_boost_db: float = Form(8.0),
    noise_percentile: float = Form(15.0),
    out_format: str = Form("wav"),
):
    """
    Multipart form-data:
    - file: audio file
    - strength: 0..1
    - residual_gate: 0..1
    - hf_boost_db: 0..20
    - noise_percentile: 5..40
    - out_format: wav | flac
    """
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    strength = _clamp(float(strength), 0.0, 1.0)
    residual_gate = _clamp(float(residual_gate), 0.0, 1.0)
    hf_boost_db = _clamp(float(hf_boost_db), 0.0, 20.0)
    noise_percentile = _clamp(float(noise_percentile), 5.0, 40.0)
    out_format = (out_format or "wav").lower().strip()
    if out_format not in ("wav", "flac"):
        out_format = "wav"

    # soundfile can read from file-like objects
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as e:
        return JSONResponse({"error": f"Unsupported audio or decode failed: {e}"}, status_code=400)

    x = np.asarray(x, dtype=np.float32)
    cfg = DenoiseConfig(
        strength=strength,
        residual_gate=residual_gate,
        hf_boost_db=hf_boost_db,
        noise_percentile=noise_percentile,
    )
    y = denoise_audio(x, int(sr), cfg)

    # write to temp file and return (FastAPI FileResponse streams it)
    suffix = ".wav" if out_format == "wav" else ".flac"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    try:
        sf.write(tmp_path, y, int(sr))
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    filename = os.path.splitext(file.filename or "audio")[0] + f"_denoised{suffix}"
    return FileResponse(
        tmp_path,
        media_type="audio/wav" if out_format == "wav" else "audio/flac",
        filename=filename,
        background=None,
    )


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Returns quick stats + suggested defaults for the UI.
    """
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Empty file"}, status_code=400)
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as e:
        return JSONResponse({"error": f"Unsupported audio or decode failed: {e}"}, status_code=400)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x_m = np.mean(x, axis=1)
    else:
        x_m = x
    rms = float(np.sqrt(np.mean(x_m * x_m) + 1e-12))
    peak = float(np.max(np.abs(x_m))) if x_m.size else 0.0

    # crude "hiss indicator": ratio of HF energy (above 6k) to total
    n_fft = 2048 if sr >= 32000 else 1024
    hop = n_fft // 4
    # short STFT
    from .denoise import stft  # local import to avoid clutter

    X, _ = stft(x_m[: min(x_m.shape[0], sr * 10)], n_fft=n_fft, hop=hop)  # first 10s max
    mag = np.abs(X).astype(np.float32)
    freqs = np.linspace(0.0, sr / 2.0, mag.shape[1], dtype=np.float32)
    hf = mag[:, freqs >= 6000.0]
    hf_ratio = float((np.mean(hf) + 1e-9) / (np.mean(mag) + 1e-9))

    # suggested params
    suggested = {
        "strength": 0.92 if rms < 0.08 else 0.85,
        "residual_gate": 0.55 if hf_ratio > 0.35 else 0.35,
        "hf_boost_db": 10.0 if hf_ratio > 0.35 else 8.0,
        "noise_percentile": 12.0 if rms < 0.08 else 15.0,
    }
    return {
        "sr": int(sr),
        "channels": 1 if x.ndim == 1 else int(x.shape[1]),
        "duration_sec": float(x_m.shape[0] / float(sr)) if sr else 0.0,
        "rms": rms,
        "peak": peak,
        "hf_ratio": hf_ratio,
        "suggested": suggested,
    }

