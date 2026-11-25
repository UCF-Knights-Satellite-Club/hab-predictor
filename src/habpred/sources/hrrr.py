"""HRRR 3 km forecasts (CONUS) from NOAA's open data bucket on AWS.

Pressure-level output (``wrfprs``) has 40 levels from 1013.2 to 50 hPa, so HRRR
only covers the flight below about 20 km. Pair it with GFS above (see
``solver.predict(upper=..., blend=...)``).

The grid is Lambert conformal; winds are grid-relative in the files and are
rotated to earth-relative here. One forecast hour downloads about 76 MB and the
default Florida crop is about 30 MB uncompressed.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..wind import GRID_LCC, WindField, lcc_forward, lcc_params, rotate_grid_to_earth
from .common import DEFAULT_BBOX, data_dir, decode, fetch_records, get, parse_idx, pressure_level
from .gfs import geopotential_to_geometric

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
VARS = ("HGT", "UGRD", "VGRD")


def file_url(cycle: datetime, fhour: int) -> str:
    return f"{BUCKET}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfprsf{fhour:02d}.grib2"


def _cache_path(cycle: datetime, fhour: int, bbox) -> Path:
    tag = "_".join(f"{b:g}" for b in bbox)
    return data_dir() / "hrrr" / f"{cycle:%Y%m%d%H}" / f"f{fhour:02d}_{tag}.nc"


def _crop_indices(proj, x0, y0, dx, dy, nx, ny, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    lats = np.linspace(lat_min, lat_max, 50)
    lons = np.linspace(lon_min, lon_max, 50)
    edge = [(la, lon_min) for la in lats] + [(la, lon_max) for la in lats] + \
           [(lat_min, lo) for lo in lons] + [(lat_max, lo) for lo in lons]
    xs, ys = zip(*[lcc_forward(proj, la, lo)[:2] for la, lo in edge])
    i0 = max(0, int(math.floor((min(xs) - x0) / dx)) - 1)
    i1 = min(nx - 1, int(math.ceil((max(xs) - x0) / dx)) + 1)
    j0 = max(0, int(math.floor((min(ys) - y0) / dy)) - 1)
    j1 = min(ny - 1, int(math.ceil((max(ys) - y0) / dy)) + 1)
    return i0, i1, j0, j1


def download_hour(cycle: datetime, fhour: int, bbox=DEFAULT_BBOX, force=False) -> WindField:
    path = _cache_path(cycle, fhour, bbox)
    if path.exists() and not force:
        return WindField.load(path)

    url = file_url(cycle, fhour)
    recs = [r for r in parse_idx(get(url + ".idx").decode())
            if r.var in VARS and pressure_level(r.level) is not None]
    fields = {}
    grid = None
    for r, msg in zip(recs, fetch_records(url, recs)):
        keys, vals = decode(msg)
        if grid is None:
            if keys["gridType"] != "lambert" or keys["Latin1InDegrees"] != keys["Latin2InDegrees"]:
                raise ValueError("unexpected HRRR grid")
            if keys.get("jScansPositively") != 1 or keys.get("iScansNegatively") != 0:
                raise ValueError("unexpected HRRR scan order")
            radius = 6371229.0 if keys.get("shapeOfTheEarth") == 6 else float(keys["radius"])
            lov = (keys["LoVInDegrees"] + 180.0) % 360.0 - 180.0
            proj = lcc_params(keys["Latin1InDegrees"], lov, radius)
            lo1 = (keys["longitudeOfFirstGridPointInDegrees"] + 180.0) % 360.0 - 180.0
            x0, y0, _ = lcc_forward(proj, keys["latitudeOfFirstGridPointInDegrees"], lo1)
            nx, ny = keys["Nx"], keys["Ny"]
            dx, dy = keys["DxInMetres"], keys["DyInMetres"]
            i0, i1, j0, j1 = _crop_indices(proj, x0, y0, dx, dy, nx, ny, bbox)
            xs = x0 + dx * np.arange(i0, i1 + 1)
            ys = y0 + dy * np.arange(j0, j1 + 1)
            theta = np.arctan2(xs[None, :], -ys[:, None])  # rotation angle per point
            grid = dict(proj=proj, xs=xs, ys=ys, theta=theta, sl=np.s_[j0:j1 + 1, i0:i1 + 1],
                        nx=nx, ny=ny, rel=keys.get("uvRelativeToGrid", 1))
        fields[(r.var, pressure_level(r.level))] = vals.reshape(grid["ny"], grid["nx"])[grid["sl"]]

    levels = sorted({p for (_, p) in fields}, reverse=True)
    levels = [p for p in levels if all((v, p) in fields for v in VARS)]
    hgt = np.stack([fields[("HGT", p)] for p in levels])
    u = np.stack([fields[("UGRD", p)] for p in levels])
    v = np.stack([fields[("VGRD", p)] for p in levels])
    if grid["rel"]:
        u, v = rotate_grid_to_earth(u, v, grid["theta"][None])
    valid = cycle + timedelta(hours=fhour)
    wf = WindField(
        kind=GRID_LCC,
        times=np.array([valid.timestamp()]),
        y=grid["ys"].astype(np.float64),
        x=grid["xs"].astype(np.float64),
        levels=np.array(levels, dtype=np.float64),
        hgt=geopotential_to_geometric(hgt)[None].astype(np.float32),
        u=u[None].astype(np.float32),
        v=v[None].astype(np.float32),
        proj=grid["proj"],
        source=f"HRRR {cycle:%Y-%m-%d %HZ}",
    )
    wf.save(path)
    return wf


def load(cycle: datetime, start: datetime, end: datetime, bbox=DEFAULT_BBOX) -> WindField:
    cycle = cycle.astimezone(timezone.utc)
    f0 = max(0, int((start - cycle).total_seconds() // 3600))
    f1 = -int(-(end - cycle).total_seconds() // 3600)
    max_f = 48 if cycle.hour % 6 == 0 else 18
    if f1 > max_f:
        raise ValueError(f"HRRR {cycle:%HZ} runs only go to f{max_f}")
    return WindField.concat_time([download_hour(cycle, h, bbox) for h in range(f0, f1 + 1)])
