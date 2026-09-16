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
  GET  /sessions            - day-grouped session list across all devices (History page)
  GET  /sessions/detail    - one representative reading for a specific session, shaped like /latest
  GET  /devices            - list all device_ids seen so far
  GET  /                   - health check
"""

import os
from typing import List, Literal, Optional

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
# Connection pool is created lazily (on first request that needs it) rather than at
# import time. Creating it eagerly here meant any transient DB outage (e.g. the Tiger
# Data free-trial auto-pause) crashed uvicorn during `import main` itself -- the whole
# service failed to boot and Render crash-looped it, taking down every endpoint,
# including ones that don't touch the DB at all. Lazy init also means the app
# self-heals once the DB comes back, without a manual redeploy: the next request
# just tries again instead of being stuck forever on a pool object created from a
# connection attempt that failed hours ago.
_db_pool = None


def get_db_pool():
    global _db_pool
    if _db_pool is not None:
        return _db_pool
    if not DATABASE_URL:
        return None
    try:
        _db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DATABASE_URL, connect_timeout=5)
    except Exception as e:
        print(f"Database connection failed (will retry on next request): {e}")
        return None
    return _db_pool

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
    # kurtosis/skewness are mathematically undefined (NaN, div-by-zero variance)
    # for a perfectly constant signal -- e.g. a stuck sensor or a mock batch that
    # forgot to add noise on one axis. Guard explicitly rather than letting NaN
    # crash JSON serialization downstream.
    if rms > 1e-9:
        kurt = float(kurtosis(sig_ac))
        sk = float(skew(sig_ac))
    else:
        kurt = 0.0
        sk = 0.0
    return {
        "rms": rms,  # g
        "crest_factor": crest,  # dimensionless (scale-invariant, unaffected by calibration)
        "kurtosis": kurt,  # dimensionless
        "skewness": sk,  # dimensionless
    }


def sanitize_json(obj):
    """Recursively replace NaN/Inf with None so json.dumps never crashes on them.
    Defense-in-depth on top of the compute_features guard above -- catches any
    other numerically-degenerate case that might slip through."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):  # obj != obj is the NaN check
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


@app.get("/")
def root():
    return {"status": "backend running", "db_connected": get_db_pool() is not None}


@app.post("/session/start")
def start_session(device_id: str, label: FaultLabel):
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
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
            pool.putconn(conn)

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

    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
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
            pool.putconn(conn)

    return {"status": "ok", "device_id": batch.device_id, **session}


@app.get("/latest")
def latest(device_id: str):
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
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
            pool.putconn(conn)

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
        return sanitize_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed formatting response for device '{device_id}': {e}")


@app.get("/latest/averaged")
def latest_averaged(device_id: str, n: int = 5):
    """
    Like /latest, but the SPECTRUM is averaged across the n most recent
    readings for this device that share the most recent one's label (so a
    recent label switch, e.g. normal -> bpfo, doesn't blend two different
    physical conditions together). The time waveform and scalar features
    (rms/crest/kurtosis/skewness) are still from the single latest reading
    only -- averaging those wouldn't mean the same thing, so this only
    smooths the thing that was actually noisy: the per-reading spectrum.

    Averages POWER (magnitude squared) across readings, then takes the square
    root -- the standard way to average magnitude spectra (this is
    Bartlett's/Welch's method for PSD estimation). Averaging the magnitudes
    directly would be a slight understatement of the result, since sqrt() is
    concave (Jensen's inequality) -- mean(sqrt(x)) <= sqrt(mean(x)).

    Each axis reports how many readings actually went into its average
    (`averaged_over`) -- normally n, but fewer right after a session start,
    or if a stray reading has a mismatched spectrum length (e.g. from a
    different sample_rate/BATCH_SIZE) and gets skipped rather than corrupting
    the average.
    """
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM readings WHERE device_id = %s ORDER BY time DESC LIMIT 1",
                (device_id,),
            )
            latest_row = cur.fetchone()
            if not latest_row:
                raise HTTPException(status_code=404, detail=f"No data yet for device '{device_id}'")
            cur.execute(
                """
                SELECT time, x_spectrum_mag, y_spectrum_mag, z_spectrum_mag
                FROM readings
                WHERE device_id = %s AND label = %s
                ORDER BY time DESC LIMIT %s
                """,
                (device_id, latest_row["label"], n),
            )
            spec_rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed for device '{device_id}' (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            pool.putconn(conn)

    latest_row = dict(latest_row)
    spec_rows = [dict(r) for r in spec_rows]

    try:
        result = {
            "timestamp": latest_row["time"].isoformat(),
            "sample_rate": latest_row["sample_rate"],
            "label": latest_row["label"],
            "trial": latest_row["trial"],
        }
        for axis in ["x", "y", "z"]:
            raw_adc = np.array(latest_row[f"{axis}_raw"], dtype=float)
            raw_g = ((raw_adc - raw_adc.mean()) * ADC_TO_G).round(5).tolist()

            freqs = latest_row[f"{axis}_spectrum_freqs"]
            expected_len = len(freqs)
            mags = [
                np.array(r[f"{axis}_spectrum_mag"], dtype=float)
                for r in spec_rows
                if r[f"{axis}_spectrum_mag"] is not None and len(r[f"{axis}_spectrum_mag"]) == expected_len
            ]
            if not mags:
                # Nothing else matched this axis's spectrum length -- fall back
                # to the latest reading's own spectrum rather than erroring out.
                mags = [np.array(latest_row[f"{axis}_spectrum_mag"], dtype=float)]
            avg_power = np.mean(np.stack(mags) ** 2, axis=0)
            avg_mag = np.sqrt(avg_power).round(6).tolist()

            result[axis] = {
                "raw": raw_g,
                "rms": latest_row[f"{axis}_rms"],
                "crest_factor": latest_row[f"{axis}_crest"],
                "kurtosis": latest_row[f"{axis}_kurtosis"],
                "skewness": latest_row[f"{axis}_skewness"],
                "spectrum_freqs": freqs,
                "spectrum_mag": avg_mag,
                "averaged_over": len(mags),
            }
        return sanitize_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed formatting response for device '{device_id}': {e}")


@app.get("/history")
def history(device_id: str, limit: int = 30):
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
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
            pool.putconn(conn)

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
    return sanitize_json(result)


@app.get("/sessions")
def get_sessions(device_id: Optional[str] = None, label: Optional[str] = None):
    """
    Day-grouped list of recorded sessions across ALL devices, for the
    dashboard's History page. One row per (day, device_id, label, trial) =
    one "session card" in the UI. Optional filters: device_id, label.
    Named /sessions (not /history) to avoid colliding with the endpoint
    above, which is a different, single-device, RMS-trend query.

    Groups by Asia/Bangkok calendar day (the dashboard shows Thai dates) --
    change the AT TIME ZONE literal if that assumption is wrong. A session
    that runs past midnight Bangkok time will be split into two day-cards;
    acceptable for this scope, not handled.
    """
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    date_trunc('day', time AT TIME ZONE 'Asia/Bangkok') AS day,
                    device_id,
                    label,
                    trial,
                    MIN(time) AS start_time,
                    MAX(time) AS end_time,
                    COUNT(*) AS batch_count,
                    AVG(x_rms) AS avg_x_rms,
                    AVG(y_rms) AS avg_y_rms,
                    AVG(z_rms) AS avg_z_rms
                FROM readings
                WHERE (%(device_id)s::text IS NULL OR device_id = %(device_id)s)
                  AND (%(label)s::text IS NULL OR label = %(label)s)
                GROUP BY day, device_id, label, trial
                ORDER BY day DESC, start_time DESC
                """,
                {"device_id": device_id, "label": label},
            )
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            pool.putconn(conn)

    days = {}
    for r in rows:
        r = dict(r)
        day_key = r["day"].date().isoformat()
        days.setdefault(day_key, []).append({
            "device_id": r["device_id"],
            "label": r["label"],
            "trial": r["trial"],
            "start_time": r["start_time"].isoformat(),
            "end_time": r["end_time"].isoformat(),
            "batch_count": r["batch_count"],
            "avg_rms": {"x": r["avg_x_rms"], "y": r["avg_y_rms"], "z": r["avg_z_rms"]},
        })

    return sanitize_json({"days": [{"date": d, "sessions": s} for d, s in sorted(days.items(), reverse=True)]})


@app.get("/sessions/detail")
def get_session_detail(device_id: str, label: str, trial: int):
    """
    One representative reading for a specific session, returned in the SAME
    shape as /latest (per-axis raw/rms/crest_factor/kurtosis/skewness/spectrum
    plus top-level sample_rate) so the dashboard's <SignalAnalysisPage> can
    render it completely unchanged -- History just fetches this and passes
    it in as the `latest` prop instead of the live reading.

    `label` is a plain str, not FaultLabel -- readings ingested with no
    active session fall back to label="unlabeled" (see /ingest), which
    isn't in that enum. Constraining it here 422'd on exactly that case.

    Picks the batch with the highest combined RMS in the session (the most
    "interesting" moment) rather than the first or last in time.
    """
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM readings
                WHERE device_id = %s AND label = %s AND trial = %s
                ORDER BY (COALESCE(x_rms, 0) + COALESCE(y_rms, 0) + COALESCE(z_rms, 0)) DESC
                LIMIT 1
                """,
                (device_id, label, trial),
            )
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed for device '{device_id}' (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            pool.putconn(conn)

    if not row:
        raise HTTPException(status_code=404, detail=f"No data for device '{device_id}', label '{label}', trial {trial}")

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
            # Same AC-only (mean-subtracted) g conversion as /latest -- keeps
            # the waveform shape identical to what SignalAnalysisPage expects.
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
        return sanitize_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed formatting response for device '{device_id}': {e}")


@app.get("/devices")
def list_devices():
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT device_id FROM readings ORDER BY device_id")
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed (check for connection pool exhaustion): {e}")
    finally:
        if conn is not None:
            pool.putconn(conn)
    return [r[0] for r in rows]