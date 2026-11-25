"""GFS 0.25 degree forecasts from NOAA's open data bucket on AWS.

Only geopotential height and u/v wind on pressure levels are downloaded, from
both the ``pgrb2`` and ``pgrb2b`` files (62 levels between 1000 and 1 hPa), and
cropped to a bounding box. One forecast hour for the default Florida box is
about 2 MB on disk; the download is about 140 MB because GRIB records are
global.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..wind import GRID_LATLON, WindField
from .common import DEFAULT_BBOX, data_dir, decode, fetch_records, get, parse_idx, pressure_level

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
VARS = ("HGT", "UGRD", "VGRD")
MIN_HPA = 1.0  # ignore the mesosphere levels (0.01-0.7 hPa)
R_EARTH = 6371008.8


def file_url(cycle: datetime, fhour: int, product: str) -> str:
    return (f"{BUCKET}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"gfs.t{cycle:%H}z.{product}.0p25.f{fhour:03d}")


def geopotential_to_geometric(h):
    """Geopotential height (gpm) to geometric altitude (m)."""
    return R_EARTH * h / (R_EARTH - h)


def _cache_path(cycle: datetime, fhour: int, bbox) -> Path:
    tag = "_".join(f"{b:g}" for b in bbox)
    return data_dir() / "gfs" / f"{cycle:%Y%m%d%H}" / f"f{fhour:03d}_{tag}.nc"


def download_hour(cycle: datetime, fhour: int, bbox=DEFAULT_BBOX, force=False) -> WindField:
    """Download (or load from cache) one GFS forecast hour cropped to ``bbox``."""
    path = _cache_path(cycle, fhour, bbox)
    if path.exists() and not force:
        return WindField.load(path)

    lat_min, lat_max, lon_min, lon_max = bbox
    fields: dict[tuple[str, float], np.ndarray] = {}
    lats = lons = None
    for product in ("pgrb2", "pgrb2b"):
        url = file_url(cycle, fhour, product)
        recs = [r for r in parse_idx(get(url + ".idx").decode())
                if r.var in VARS and (p := pressure_level(r.level)) is not None and p >= MIN_HPA
                and (r.var, p) not in fields]
        for r, msg in zip(recs, fetch_records(url, recs)):
            keys, vals = decode(msg)
            ni, nj = keys["Ni"], keys["Nj"]
            grid = vals.reshape(nj, ni)
            lat_axis = np.linspace(keys["latitudeOfFirstGridPointInDegrees"],
                                   keys["latitudeOfLastGridPointInDegrees"], nj)
            lon_axis = keys["longitudeOfFirstGridPointInDegrees"] + \
                keys["iDirectionIncrementInDegrees"] * np.arange(ni)
            lon_axis = (lon_axis + 180.0) % 360.0 - 180.0
            yi = np.where((lat_axis >= lat_min - 1e-6) & (lat_axis <= lat_max + 1e-6))[0]
            xi = np.where((lon_axis >= lon_min - 1e-6) & (lon_axis <= lon_max + 1e-6))[0]
            if lats is None:
                lats, lons = lat_axis[yi], lon_axis[xi]
            fields[(r.var, pressure_level(r.level))] = grid[np.ix_(yi, xi)]

    levels = sorted({p for (_, p) in fields}, reverse=True)
    levels = [p for p in levels if all((v, p) in fields for v in VARS)]
    order_y = np.argsort(lats)
    order_x = np.argsort(lons)

    def stack(var):
        a = np.stack([fields[(var, p)] for p in levels])
        return a[:, order_y][:, :, order_x][None].astype(np.float32)

    valid = cycle + timedelta(hours=fhour)
    wf = WindField(
        kind=GRID_LATLON,
        times=np.array([valid.timestamp()]),
        y=lats[order_y].astype(np.float64),
        x=lons[order_x].astype(np.float64),
        levels=np.array(levels, dtype=np.float64),
        hgt=geopotential_to_geometric(stack("HGT")).astype(np.float32),
        u=stack("UGRD"),
        v=stack("VGRD"),
        source=f"GFS 0.25 {cycle:%Y-%m-%d %HZ}",
    )
    wf.save(path)
    return wf


def load(cycle: datetime, start: datetime, end: datetime, bbox=DEFAULT_BBOX,
         step_hours: int = 1) -> WindField:
    """GFS winds from ``cycle`` covering valid times ``start``..``end`` (inclusive, padded
    to whole forecast hours)."""
    cycle = cycle.astimezone(timezone.utc)
    f0 = int((start - cycle).total_seconds() // 3600)
    f1 = -int(-(end - cycle).total_seconds() // 3600)
    f0 = max(0, f0 - f0 % step_hours)
    f1 = f1 + (-f1) % step_hours
    if f1 > 120 and step_hours == 1:
        raise ValueError("hourly GFS output only goes to f120; use step_hours=3")
    hours = range(f0, f1 + 1, step_hours)
    return WindField.concat_time([download_hour(cycle, h, bbox) for h in hours])


def latest_cycle(now: datetime | None = None, max_back: int = 6) -> datetime:
    """Most recent cycle whose f000 index exists on AWS."""
    now = now or datetime.now(timezone.utc)
    c = now.replace(minute=0, second=0, microsecond=0)
    c -= timedelta(hours=c.hour % 6)
    for _ in range(max_back):
        try:
            get(file_url(c, 120, "pgrb2b") + ".idx")
            return c
        except FileNotFoundError:
            c -= timedelta(hours=6)
    raise RuntimeError("no recent GFS cycle found")
