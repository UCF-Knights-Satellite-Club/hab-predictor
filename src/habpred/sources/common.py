"""Shared helpers for downloading GRIB2 records by byte range using .idx files."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import eccodes
import numpy as np
import requests

USER_AGENT = "habpred/0.1 (+https://github.com/UCF-Knights-Satellite-Club/hab-predictor)"

# Florida plus margin for long winter drifts.
DEFAULT_BBOX = (22.0, 34.0, -90.0, -76.0)  # lat_min, lat_max, lon_min, lon_max


def data_dir() -> Path:
    return Path(os.environ.get("HABPRED_DATA", Path(__file__).resolve().parents[3] / "data"))


_session = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        _session = s
    return _session


def get(url: str, headers: dict | None = None, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            r = session().get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


@dataclass
class IdxRecord:
    num: int
    start: int
    end: int | None  # inclusive, None = to end of file
    var: str
    level: str
    fcst: str


def parse_idx(text: str) -> list[IdxRecord]:
    lines = [l.split(":") for l in text.strip().splitlines()]
    recs = []
    for i, f in enumerate(lines):
        start = int(f[1])
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        recs.append(IdxRecord(int(f[0].split(".")[0]), start, end, f[3], f[4], f[5]))
    return recs


def pressure_level(level: str) -> float | None:
    """'500 mb' -> 500.0, anything else -> None."""
    if level.endswith(" mb"):
        try:
            return float(level[:-3])
        except ValueError:
            return None
    return None


def fetch_records(url: str, recs: list[IdxRecord], workers: int = 16) -> list[bytes]:
    def one(r: IdxRecord) -> bytes:
        rng = f"bytes={r.start}-" + ("" if r.end is None else str(r.end))
        return get(url, headers={"Range": rng})

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, recs))


def decode(msg: bytes) -> tuple[dict, np.ndarray]:
    gid = eccodes.codes_new_from_message(msg)
    try:
        keys = {}
        for k in ("Ni", "Nj", "Nx", "Ny", "gridType", "latitudeOfFirstGridPointInDegrees",
                  "longitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
                  "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees",
                  "LoVInDegrees", "Latin1InDegrees", "Latin2InDegrees", "DxInMetres",
                  "DyInMetres", "shapeOfTheEarth", "radius", "uvRelativeToGrid",
                  "jScansPositively", "iScansNegatively", "shortName", "level",
                  "typeOfLevel", "validityDate", "validityTime"):
            try:
                keys[k] = eccodes.codes_get(gid, k)
            except eccodes.KeyValueNotFoundError:
                pass
        vals = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)
    return keys, vals
