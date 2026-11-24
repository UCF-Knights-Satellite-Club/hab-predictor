"""Radiosonde flights from the SondeHub history archive (CC BY-SA 2.0).

The archive is an S3 bucket served at history.sondehub.org:

* ``launchsites/{site}/{YYYY}/{MM}/{DD}/{serial}.json``: a few summary frames per
  flight (first heard, highest, last heard);
* ``serial/{serial}.json.gz``: every received frame, often duplicated by several
  receivers.

Tracks are thinned to one row per frame and stored as gzipped CSV.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .common import data_dir, get

BASE = "https://history.sondehub.org"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Florida NWS / range launch sites (SondeHub launch-site ids are WMO numbers)
FLORIDA_SITES = {
    "72210": "Tampa Bay (Ruskin)",
    "74794": "Cape Canaveral",
    "72206": "Jacksonville",
    "72202": "Miami",
}


def list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{BASE}/?list-type=2&prefix={prefix}"
        if token:
            from urllib.parse import quote
            url += "&continuation-token=" + quote(token, safe="")
        root = ET.fromstring(get(url))
        keys += [c.find(NS + "Key").text for c in root.findall(NS + "Contents")]
        if root.findtext(NS + "IsTruncated") != "true":
            return keys
        token = root.findtext(NS + "NextContinuationToken")


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_summary(key: str) -> dict:
    frames = json.loads(get(f"{BASE}/{key}"))
    frames.sort(key=lambda f: f["datetime"])
    first, last = frames[0], frames[-1]
    top = max(frames, key=lambda f: f.get("alt", 0))
    return {
        "serial": first["serial"],
        "site": first.get("launch_site"),
        "type": first.get("type"),
        "first_time": first["datetime"], "first_lat": first["lat"], "first_lon": first["lon"],
        "first_alt": first["alt"],
        "top_time": top["datetime"], "top_lat": top["lat"], "top_lon": top["lon"],
        "top_alt": top["alt"],
        "last_time": last["datetime"], "last_lat": last["lat"], "last_lon": last["lon"],
        "last_alt": last["alt"],
    }


def summaries(sites, months: list[tuple[int, int]], workers: int = 16) -> list[dict]:
    """Flight summaries for the given sites and (year, month) pairs."""
    keys = []
    for site in sites:
        for y, m in months:
            keys += list_keys(f"launchsites/{site}/{y:04d}/{m:02d}/")
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(fetch_summary, keys))
    return out


def track_path(serial: str) -> Path:
    return data_dir() / "sondes" / "tracks" / f"{serial}.csv.gz"


def fetch_track(serial: str, force: bool = False) -> Path:
    """Download and thin one flight's full track. Returns the CSV path."""
    path = track_path(serial)
    if path.exists() and not force:
        return path
    raw = get(f"{BASE}/serial/{serial}.json.gz")
    try:
        frames = json.loads(gzip.decompress(raw))
    except OSError:  # served already decompressed
        frames = json.loads(raw)
    rows = {}
    for f in frames:
        if "lat" not in f or "lon" not in f or "alt" not in f:
            continue
        t = f["datetime"]
        if t not in rows:
            rows[t] = (t, f["lat"], f["lon"], f["alt"], f.get("vel_v", ""), f.get("vel_h", ""),
                       f.get("heading", ""), f.get("pressure", ""), f.get("temp", ""))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "lat", "lon", "alt", "vel_v", "vel_h", "heading", "pressure", "temp"])
    for k in sorted(rows):
        w.writerow(rows[k])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(gzip.compress(buf.getvalue().encode()))
    tmp.replace(path)
    return path


def load_track(serial: str):
    """Return (unix time, lat, lon, alt) numpy arrays for a downloaded track."""
    import numpy as np
    with gzip.open(track_path(serial), "rt") as fh:
        r = list(csv.DictReader(fh))
    t = np.array([_t(x["time"]).timestamp() for x in r])
    return (t, np.array([float(x["lat"]) for x in r]), np.array([float(x["lon"]) for x in r]),
            np.array([float(x["alt"]) for x in r]))
