"""
Motor vibration monitoring backend (Phase 1: no RF model yet).
Receives raw ADXL335 batches from ESP32, computes FFT + time-domain features,
stores to Firestore, and serves the dashboard.

Endpoints:
  POST /ingest   - ESP32 posts a batch here
  GET  /latest   - dashboard polls this for current reading + spectrum
  GET  /history  - dashboard polls this for RMS trend
  GET  /         - health check
"""

import json
import os
from datetime import datetime, timezone
from typing import List

import firebase_admin
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from pydantic import BaseModel
from scipy.stats import kurtosis, skew

app = FastAPI(title="Motor Vibration Monitor")

# CORS: allow the dashboard to call this API. Tighten allow_origins to your
# actual Vercel domain once deployed instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Firebase init ---
# Set env var FIREBASE_CREDENTIALS on Render to the full contents of your
# service account JSON key (as a single-line string).
_cred_json = os.environ.get("FIREBASE_CREDENTIALS")
if _cred_json:
    cred = credentials.Certificate(json.loads(_cred_json))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    db = None  # allows the app to boot locally without Firebase for a quick syntax check


class Batch(BaseModel):
    x: List[int]
    y: List[int]
    z: List[int]
    sample_rate: int


def compute_fft(signal, fs):
    sig = np.array(signal, dtype=float)
    sig = sig - np.mean(sig)
    window = np.hanning(len(sig))
    windowed = sig * window
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
    magnitude = np.abs(fft_vals) * 2 / np.sum(window)
    return freqs.tolist(), magnitude.tolist()


def compute_features(signal):
    sig = np.array(signal, dtype=float)
    sig_ac = sig - np.mean(sig)
    rms = float(np.sqrt(np.mean(sig_ac ** 2)))
    peak = float(np.max(np.abs(sig_ac)))
    crest = peak / rms if rms > 0 else 0.0
    return {
        "rms": rms,
        "crest_factor": crest,
        "kurtosis": float(kurtosis(sig_ac)),
        "skewness": float(skew(sig_ac)),
    }


@app.get("/")
def root():
    return {"status": "backend running", "firebase_connected": db is not None}


@app.post("/ingest")
def ingest(batch: Batch):
    if db is None:
        raise HTTPException(status_code=503, detail="Firebase not configured")

    timestamp = datetime.now(timezone.utc)

    # 1) store raw batch (kept for future sensor-swap / reprocessing flexibility)
    db.collection("raw_batches").add({
        "x": batch.x, "y": batch.y, "z": batch.z,
        "sample_rate": batch.sample_rate,
        "timestamp": timestamp,
    })

    # 2) compute features + spectrum per axis
    reading = {"timestamp": timestamp, "sample_rate": batch.sample_rate}
    stats_row = {"timestamp": timestamp}
    for axis, data in [("x", batch.x), ("y", batch.y), ("z", batch.z)]:
        feats = compute_features(data)
        freqs, mag = compute_fft(data, batch.sample_rate)
        reading[axis] = {"raw": data, **feats, "spectrum_freqs": freqs, "spectrum_mag": mag}
        stats_row[f"rms_{axis}"] = feats["rms"]

    # "latest/reading" is a single doc, overwritten every ingest -- dashboard reads this
    db.collection("latest").document("reading").set(reading)

    # "computed_stats" grows over time -- lightweight, used for the trend chart
    db.collection("computed_stats").add(stats_row)

    return {"status": "ok"}


@app.get("/latest")
def latest():
    if db is None:
        raise HTTPException(status_code=503, detail="Firebase not configured")
    doc = db.collection("latest").document("reading").get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="No data yet -- has the ESP32 sent a batch?")
    data = doc.to_dict()
    data["timestamp"] = data["timestamp"].isoformat()
    return data


@app.get("/history")
def history(limit: int = 50):
    if db is None:
        raise HTTPException(status_code=503, detail="Firebase not configured")
    docs = (
        db.collection("computed_stats")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    rows = []
    for doc in docs:
        d = doc.to_dict()
        rows.append({
            "timestamp": d["timestamp"].isoformat(),
            "rms_x": d.get("rms_x"),
            "rms_y": d.get("rms_y"),
            "rms_z": d.get("rms_z"),
        })
    rows.reverse()
    return rows