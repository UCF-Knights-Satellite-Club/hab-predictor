"""Client for the SondeHub-hosted Tawhiri predictor (for comparison only).

SondeHub asks that its API see limited use from third-party tools, so call it
sparingly (a few requests per comparison run).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .solver import Flight
from .sources.common import session

API = "https://api.v2.sondehub.org/tawhiri"


def predict(launch_time: datetime, lat: float, lon: float, launch_alt: float,
            ascent_rate: float, burst_alt: float, descent_rate: float) -> tuple[Flight, dict]:
    params = {
        "profile": "standard_profile",
        "launch_latitude": lat,
        "launch_longitude": lon % 360.0,
        "launch_datetime": launch_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "launch_altitude": launch_alt,
        "ascent_rate": ascent_rate,
        "burst_altitude": burst_alt,
        "descent_rate": descent_rate,
    }
    r = session().get(API, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    if "error" in js:
        raise RuntimeError(js["error"])
    pts = []
    burst_idx = -1
    for stage in js["prediction"]:
        traj = stage["trajectory"]
        if stage["stage"] == "descent":
            burst_idx = len(pts) - 1
            traj = traj[1:]  # first descent point repeats the burst point
        for p in traj:
            t = datetime.fromisoformat(p["datetime"].replace("Z", "+00:00")).timestamp()
            pts.append((t, p["latitude"], (p["longitude"] + 180.0) % 360.0 - 180.0, p["altitude"]))
    a = np.array(pts)
    return Flight(a[:, 0], a[:, 1], a[:, 2], a[:, 3], burst_idx, 0), js["request"]
