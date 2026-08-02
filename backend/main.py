"""
Motor vibration monitoring backend (Phase 1: no RF model yet).
Backed by TimescaleDB (PostgreSQL) with multi-device support so several ESP32
units can stream concurrently (e.g. simulating multiple motors in a factory).

Endpoints:
  POST /session/start    - mark start of a labeled recording session for ONE device
  GET  /session/current  - current label/trial for one device
  POST /ingest            - ESP32 posts a batch here (must include device_id)
  GET  /latest             - latest reading for one device
  GET  /history            - RMS trend for one device
  GET  /devices            - list all device_ids seen so far
  GET  /                   - health check
"""

import os
from typing import List, Literal

import numpy as np
import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import kurtosis, skew

app = FastAPI(title="Motor Vibration Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set env var DATABASE_URL on Render to your Timescale Cloud connection string,
# e.g. postgresql://user:pass@host:port/dbname?sslmode=require
DATABASE_URL = os.environ.get("DATABASE_URL")
db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DATABASE_URL) if DATABASE_URL else None

FaultLabel = Literal["normal", "bpfo", "bpfi", "ftf", "bsf", "monitoring"]

# ADXL335 calibration: converts raw ESP32 ADC counts (0-4095, 12-bit) to g.
# Nominal values from the ADXL335 datasheet at 3.3V supply -- approximate, not
# lab-calibrated per unit (manufacturing tolerance + ESP32 ADC nonlinearity mean
# this is good for relative comparison, not instrument-grade absolute accuracy).
ADC_MAX = 4095
ADC_VREF = 3.3
SENSITIVITY_V_PER_G = 0.33
ADC_TO_G = (ADC_VREF / ADC_MAX) / SENSITIVITY_V_PER_G

# Per-device current recording session (label + trial), kept in memory for speed.
# Keyed by device_id -- multiple ESP32 units can each be in a different state at
# the same time (e.g. motor 1 = normal, motor 2 = bpfo, tested concurrently).
_current_sessions = {}


class Batch(BaseModel):
    device_id: str
    x: List[int]
    y: List[int]
    z: List[int]
    sample_rate: int


def compute_fft(signal, fs):
    sig = np.array(signal, dtype=float) * ADC_TO_G
    sig = sig - np.mean(sig)
    window = np.hanning(len(sig))
    windowed = sig * window
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
    magnitude = np.abs(fft_vals) * 2 / np.sum(window)
    return freqs.tolist(), magnitude.tolist()


def compute_features(signal):
    sig = np.array(signal, dtype=float) * ADC_TO_G  # ADC counts -> g
    sig_ac = sig - np.mean(sig)
    rms = float(np.sqrt(np.mean(sig_ac ** 2)))
    peak = float(np.max(np.abs(sig_ac)))
    crest = peak / rms if rms > 0 else 0.0
    return {
        "rms": rms,  # g
        "crest_factor": crest,  # dimensionless (scale-invariant, unaffected by calibration)
        "kurtosis": float(kurtosis(sig_ac)),  # dimensionless
        "skewness": float(skew(sig_ac)),  # dimensionless
    }


@app.get("/")
def root():
    return {"status": "backend running", "db_connected": db_pool is not None}


@app.post("/session/start")
def start_session(device_id: str, label: FaultLabel):
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(trial), 0) FROM readings WHERE device_id = %s AND label = %s",
                (device_id, label),
            )
            max_trial = cur.fetchone()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed (check for a missing index/table, or connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            db_pool.putconn(conn)

    _current_sessions[device_id] = {"label": label, "trial": max_trial + 1}
    return {"status": "ok", "device_id": device_id, **_current_sessions[device_id]}


@app.get("/session/current")
def get_current_session(device_id: str):
    return _current_sessions.get(device_id, {"label": "unlabeled", "trial": 0})


@app.post("/ingest")
def ingest(batch: Batch):
    session = _current_sessions.get(batch.device_id, {"label": "unlabeled", "trial": 0})

    feats = {}
    for axis, data in [("x", batch.x), ("y", batch.y), ("z", batch.z)]:
        f = compute_features(data)
        freqs, mag = compute_fft(data, batch.sample_rate)
        feats[axis] = {**f, "freqs": freqs, "mag": mag}

    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO readings (
                    device_id, label, trial, sample_rate, x_raw, y_raw, z_raw,
                    x_rms, x_crest, x_kurtosis, x_skewness, x_spectrum_freqs, x_spectrum_mag,
                    y_rms, y_crest, y_kurtosis, y_skewness, y_spectrum_freqs, y_spectrum_mag,
                    z_rms, z_crest, z_kurtosis, z_skewness, z_spectrum_freqs, z_spectrum_mag
                ) VALUES (%s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s)
                """,
                (
                    batch.device_id, session["label"], session["trial"], batch.sample_rate,
                    batch.x, batch.y, batch.z,
                    feats["x"]["rms"], feats["x"]["crest_factor"], feats["x"]["kurtosis"], feats["x"]["skewness"],
                    feats["x"]["freqs"], feats["x"]["mag"],
                    feats["y"]["rms"], feats["y"]["crest_factor"], feats["y"]["kurtosis"], feats["y"]["skewness"],
                    feats["y"]["freqs"], feats["y"]["mag"],
                    feats["z"]["rms"], feats["z"]["crest_factor"], feats["z"]["kurtosis"], feats["z"]["skewness"],
                    feats["z"]["freqs"], feats["z"]["mag"],
                ),
            )
            conn.commit()
    except Exception as e:
        if conn is not None:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Insert failed (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            db_pool.putconn(conn)

    return {"status": "ok", "device_id": batch.device_id, **session}


@app.get("/latest")
def latest(device_id: str):
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM readings WHERE device_id = %s ORDER BY time DESC LIMIT 1",
                (device_id,),
            )
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed for device '{device_id}' (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            db_pool.putconn(conn)

    if not row:
        raise HTTPException(status_code=404, detail=f"No data yet for device '{device_id}'")

    try:
        row = dict(row)
        result = {
            "timestamp": row["time"].isoformat(),
            "sample_rate": row["sample_rate"],
            "label": row["label"],
            "trial": row["trial"],
        }
        for axis in ["x", "y", "z"]:
            raw_adc = np.array(row[f"{axis}_raw"], dtype=float)
            # AC-only (mean-subtracted) waveform in g -- shows vibration fluctuation,
            # not absolute tilt, so it doesn't require knowing this unit's exact
            # zero-g offset voltage (which isn't calibrated per-device here).
            raw_g = ((raw_adc - raw_adc.mean()) * ADC_TO_G).round(5).tolist()
            result[axis] = {
                "raw": raw_g,
                "rms": row[f"{axis}_rms"],
                "crest_factor": row[f"{axis}_crest"],
                "kurtosis": row[f"{axis}_kurtosis"],
                "skewness": row[f"{axis}_skewness"],
                "spectrum_freqs": row[f"{axis}_spectrum_freqs"],
                "spectrum_mag": row[f"{axis}_spectrum_mag"],
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed formatting response for device '{device_id}': {e}")


@app.get("/history")
def history(device_id: str, limit: int = 30):
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT time, x_rms, y_rms, z_rms, label, trial
                FROM readings WHERE device_id = %s
                ORDER BY time DESC LIMIT %s
                """,
                (device_id, limit),
            )
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed for device '{device_id}' (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            db_pool.putconn(conn)

    result = [
        {
            "timestamp": r["time"].isoformat(),
            "rms_x": r["x_rms"],
            "rms_y": r["y_rms"],
            "rms_z": r["z_rms"],
            "label": r["label"],
            "trial": r["trial"],
        }
        for r in rows
    ]
    result.reverse()
    return result


@app.get("/devices")
def list_devices():
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT device_id FROM readings ORDER BY device_id")
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            db_pool.putconn(conn)
    return [r[0] for r in rows]