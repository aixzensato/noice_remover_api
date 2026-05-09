from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    return x


def peak_normalize(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 1e-9:
        return x
    g = min(1.0, float(peak) / m)
    return (x * g).astype(np.float32, copy=False)


def stft(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32, copy=False)
    if x.ndim != 1:
        raise ValueError("STFT expects mono 1D array")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")

    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (x.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    X = np.fft.rfft(frames * win[None, :], axis=1)
    return X, win


def istft(X: np.ndarray, win: np.ndarray, hop: int, length: int) -> np.ndarray:
    n_frames, n_bins = X.shape
    n_fft = (n_bins - 1) * 2
    y_len = n_fft + (n_frames - 1) * hop
    y = np.zeros((y_len,), dtype=np.float32)
    wsum = np.zeros((y_len,), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = np.fft.irfft(X[i], n=n_fft).astype(np.float32)
        y[start : start + n_fft] += frame * win
        wsum[start : start + n_fft] += win * win
    nz = wsum > 1e-8
    y[nz] /= wsum[nz]
    return y[:length]


@dataclass(frozen=True)
class DenoiseConfig:
    strength: float = 0.85  # 0..1
    noise_percentile: float = 15.0  # lower = more "noise-only" frames
    hf_boost_db: float = 8.0  # extra suppression for high-freq hiss
    residual_gate: float = 0.35  # 0..1 (2nd-stage gate intensity)
    gate_snr_db: float = 3.0

    # smoothing
    attack: float = 0.60
    release: float = 0.98

    # keep some noise to avoid musical noise
    floor_min: float = 0.01
    floor_max: float = 0.12


def denoise_mono(x: np.ndarray, sr: int, cfg: DenoiseConfig) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - float(np.mean(x))  # DC remove

    n_fft = 2048 if sr >= 32000 else 1024
    hop = n_fft // 4
    eps = 1e-10

    X, win = stft(x, n_fft=n_fft, hop=hop)
    mag = np.abs(X).astype(np.float32)
    pow_ = mag * mag

    frame_energy = np.mean(pow_, axis=1)
    thr = np.percentile(frame_energy, float(cfg.noise_percentile))
    noise_frames = pow_[frame_energy <= thr]
    if noise_frames.shape[0] < 4:
        n0 = max(4, int((0.25 * sr) // hop))
        noise_frames = pow_[: min(n0, pow_.shape[0])]
    noise_psd = np.median(noise_frames, axis=0)

    # Frequency-shaped suppression: more aggressive on high frequencies (hiss).
    freqs = np.linspace(0.0, sr / 2.0, noise_psd.shape[0], dtype=np.float32)
    hf_w = (freqs / (sr / 2.0 + 1e-9)) ** 1.3
    hf_boost = 10 ** (float(cfg.hf_boost_db) / 20.0)
    noise_psd_shaped = noise_psd * (1.0 + hf_w * (hf_boost - 1.0))

    s = float(max(0.0, min(1.0, cfg.strength)))
    alpha = 1.5 + 8.0 * s
    floor = 0.10 - 0.07 * s
    floor = float(max(cfg.floor_min, min(cfg.floor_max, floor)))

    snr = pow_ / (noise_psd_shaped[None, :] + eps)
    gain = 1.0 - alpha / (snr + alpha)
    gain = np.clip(gain, floor, 1.0).astype(np.float32)

    # attack/release smoothing
    atk = float(cfg.attack)
    rel = float(cfg.release)
    g_sm = np.empty_like(gain)
    g_prev = gain[0]
    g_sm[0] = g_prev
    for i in range(1, gain.shape[0]):
        g = gain[i]
        a = atk if np.mean(g) > np.mean(g_prev) else rel
        g_prev = a * g_prev + (1.0 - a) * g
        g_sm[i] = g_prev

    # Stage 1 output
    Y = X * g_sm.astype(np.complex64)

    # Stage 2 residual gate
    rg = float(max(0.0, min(1.0, cfg.residual_gate)))
    if rg > 1e-6:
        snr_db = 10.0 * np.log10(snr + eps)
        thr_db = float(cfg.gate_snr_db) + (6.0 * (1.0 - s))
        t = (snr_db - thr_db) / 6.0
        gate = 1.0 / (1.0 + np.exp(-t))
        min_gate = 0.15 - 0.10 * s
        min_gate = float(max(0.02, min(0.25, min_gate)))
        gate = (min_gate + (1.0 - min_gate) * gate).astype(np.float32)
        g2 = (1.0 - rg) * 1.0 + rg * gate
        Y = Y * g2.astype(np.complex64)

    y = istft(Y, win, hop=hop, length=x.shape[0])
    return y.astype(np.float32, copy=False)


def denoise_audio(x: np.ndarray, sr: int, cfg: DenoiseConfig) -> np.ndarray:
    if x.ndim == 1:
        return peak_normalize(denoise_mono(x, sr, cfg))
    x2 = ensure_2d(x)
    chs = [denoise_mono(x2[:, ch], sr, cfg) for ch in range(x2.shape[1])]
    y = np.stack(chs, axis=1).astype(np.float32)
    return peak_normalize(y)

