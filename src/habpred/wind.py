"""Gridded wind fields and fast 4-D interpolation.

A :class:`WindField` holds height, u and v on pressure levels, shaped
``[time, level, y, x]``. Levels are ordered bottom to top (pressure decreasing,
height increasing). Two horizontal grids are supported:

* ``GRID_LATLON``: ``y`` is latitude and ``x`` is longitude, both ascending, in
  degrees (GFS, ECMWF, ERA5).
* ``GRID_LCC``: ``y`` and ``x`` are metres on a spherical Lambert conformal
  conic projection described by ``proj`` (HRRR). Winds must already be rotated
  to earth-relative u/v.

Interpolation follows Tawhiri: trilinear in (time, y, x) for each level, a
binary search on the interpolated heights to find the levels bracketing the
requested altitude, then linear in altitude. Below the lowest level and above
the highest level the wind is held constant rather than extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit

GRID_LATLON = 0
GRID_LCC = 1

# Status codes returned by the interpolator and solver.
OK = 0
OUT_OF_DOMAIN = 1  # outside the grid in time or space


@dataclass
class WindField:
    kind: int
    times: np.ndarray  # float64 unix seconds, ascending
    y: np.ndarray  # float64, ascending
    x: np.ndarray  # float64, ascending
    levels: np.ndarray  # float64 hPa, bottom to top
    hgt: np.ndarray  # float32 [t, lev, y, x], metres
    u: np.ndarray  # float32 [t, lev, y, x], m/s east
    v: np.ndarray  # float32 [t, lev, y, x], m/s north
    proj: np.ndarray = field(default_factory=lambda: np.zeros(8))  # see lcc_params
    source: str = ""

    def as_tuple(self):
        """The numba-friendly representation passed to the solver."""
        return (
            self.kind,
            np.ascontiguousarray(self.proj, dtype=np.float64),
            np.ascontiguousarray(self.times, dtype=np.float64),
            np.ascontiguousarray(self.y, dtype=np.float64),
            np.ascontiguousarray(self.x, dtype=np.float64),
            np.ascontiguousarray(self.hgt, dtype=np.float32),
            np.ascontiguousarray(self.u, dtype=np.float32),
            np.ascontiguousarray(self.v, dtype=np.float32),
        )

    # ----- persistence -------------------------------------------------
    def to_dataset(self) -> xr.Dataset:
        dims = ("time", "level", "y", "x")
        ds = xr.Dataset(
            {"hgt": (dims, self.hgt), "u": (dims, self.u), "v": (dims, self.v)},
            coords={"time": self.times, "level": self.levels, "y": self.y, "x": self.x},
            attrs={"kind": self.kind, "proj": self.proj.tolist(), "source": self.source},
        )
        return ds

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        enc = {k: {"compression": "gzip", "compression_opts": 4} for k in ("hgt", "u", "v")}
        tmp = path.with_suffix(".tmp")
        self.to_dataset().to_netcdf(tmp, engine="h5netcdf", encoding=enc)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "WindField":
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            ds = ds.load()
        return cls(
            kind=int(ds.attrs["kind"]),
            times=ds["time"].values.astype(np.float64),
            y=ds["y"].values.astype(np.float64),
            x=ds["x"].values.astype(np.float64),
            levels=ds["level"].values.astype(np.float64),
            hgt=ds["hgt"].values.astype(np.float32),
            u=ds["u"].values.astype(np.float32),
            v=ds["v"].values.astype(np.float32),
            proj=np.asarray(ds.attrs["proj"], dtype=np.float64),
            source=str(ds.attrs.get("source", "")),
        )

    @classmethod
    def concat_time(cls, fields: list["WindField"]) -> "WindField":
        """Join single-time fields (same grid and levels) along time."""
        fields = sorted(fields, key=lambda f: f.times[0])
        f0 = fields[0]
        for f in fields[1:]:
            if not (np.array_equal(f.levels, f0.levels) and np.array_equal(f.x, f0.x)
                    and np.array_equal(f.y, f0.y)):
                raise ValueError("fields have different grids or levels")
        return cls(
            kind=f0.kind,
            times=np.concatenate([f.times for f in fields]),
            y=f0.y, x=f0.x, levels=f0.levels,
            hgt=np.concatenate([f.hgt for f in fields]),
            u=np.concatenate([f.u for f in fields]),
            v=np.concatenate([f.v for f in fields]),
            proj=f0.proj, source=f0.source,
        )

    def subsample(self, step_xy: int = 1, step_t: int = 1) -> "WindField":
        """Coarsen the grid by taking every n-th point (e.g. 0.25 deg -> 0.5 deg)."""
        s = (slice(None, None, step_t), slice(None), slice(None, None, step_xy),
             slice(None, None, step_xy))
        return WindField(self.kind, self.times[::step_t], self.y[::step_xy], self.x[::step_xy],
                         self.levels, self.hgt[s], self.u[s], self.v[s], self.proj,
                         self.source + f" (every {step_xy} pts, {step_t} times)")


# ----- Lambert conformal conic ------------------------------------------

def lcc_params(lat1_deg: float, lon0_deg: float, radius: float) -> np.ndarray:
    """Pack spherical tangent-cone LCC parameters: [n, F*R, lon0_rad, R, 0...]."""
    phi1 = math.radians(lat1_deg)
    n = math.sin(phi1)
    f = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** n / n
    return np.array([n, f * radius, math.radians(lon0_deg), radius, 0, 0, 0, 0], dtype=np.float64)


@njit(cache=True)
def lcc_forward(proj, lat_deg, lon_deg):
    """Lat/lon (deg) -> projected (x, y) metres. Also returns the rotation angle theta."""
    n = proj[0]
    fr = proj[1]
    lon0 = proj[2]
    phi = math.radians(lat_deg)
    dlon = math.radians(lon_deg) - lon0
    # wrap to [-pi, pi]
    dlon = (dlon + math.pi) % (2 * math.pi) - math.pi
    rho = fr / math.tan(math.pi / 4 + phi / 2) ** n
    theta = n * dlon
    return rho * math.sin(theta), -rho * math.cos(theta), theta


def rotate_grid_to_earth(u_g, v_g, theta):
    """Rotate LCC grid-relative winds to earth-relative (east, north)."""
    c, s = np.cos(theta), np.sin(theta)
    return u_g * c + v_g * s, -u_g * s + v_g * c


# ----- interpolation kernels ------------------------------------------

@njit(cache=True)
def _bracket(axis, x):
    n = axis.shape[0]
    if not (x >= axis[0] and x <= axis[n - 1]):
        return -1, 0.0
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if axis[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo, (x - axis[lo]) / (axis[hi] - axis[lo])


@njit(cache=True)
def _tri(arr, it, iy, ix, lev, ft, fy, fx):
    """Trilinear interpolation of arr[t, lev, y, x] at a fixed level."""
    nt = arr.shape[0]
    it1 = it + 1 if it + 1 < nt else it
    a00 = arr[it, lev, iy, ix] * (1 - fx) + arr[it, lev, iy, ix + 1] * fx
    a01 = arr[it, lev, iy + 1, ix] * (1 - fx) + arr[it, lev, iy + 1, ix + 1] * fx
    a10 = arr[it1, lev, iy, ix] * (1 - fx) + arr[it1, lev, iy, ix + 1] * fx
    a11 = arr[it1, lev, iy + 1, ix] * (1 - fx) + arr[it1, lev, iy + 1, ix + 1] * fx
    a0 = a00 * (1 - fy) + a01 * fy
    a1 = a10 * (1 - fy) + a11 * fy
    return a0 * (1 - ft) + a1 * ft


@njit(cache=True)
def wind_at(fld, t, lat, lon, alt):
    """Interpolate (u, v) at time t (unix s), lat/lon (deg) and altitude (m).

    Returns (u, v, status).
    """
    kind, proj, times, ys, xs, hgt, u, v = fld
    if times.shape[0] == 1:
        if t != times[0]:
            return 0.0, 0.0, OUT_OF_DOMAIN
        it, ft = 0, 0.0
    else:
        it, ft = _bracket(times, t)
    if kind == GRID_LATLON:
        yq = lat
        xq = lon
    else:
        xq, yq, _ = lcc_forward(proj, lat, lon)
    iy, fy = _bracket(ys, yq)
    ix, fx = _bracket(xs, xq)
    if it < 0 or iy < 0 or ix < 0:
        return 0.0, 0.0, OUT_OF_DOMAIN
    if iy == ys.shape[0] - 1:
        iy -= 1
        fy = 1.0
    if ix == xs.shape[0] - 1:
        ix -= 1
        fx = 1.0

    nlev = hgt.shape[1]
    h_bot = _tri(hgt, it, iy, ix, 0, ft, fy, fx)
    h_top = _tri(hgt, it, iy, ix, nlev - 1, ft, fy, fx)
    if alt <= h_bot:
        lo, w = 0, 0.0
    elif alt >= h_top:
        lo, w = nlev - 2, 1.0
    else:
        lo = 0
        hi = nlev - 1
        h_lo = h_bot
        h_hi = h_top
        while hi - lo > 1:
            mid = (lo + hi) // 2
            h_mid = _tri(hgt, it, iy, ix, mid, ft, fy, fx)
            if h_mid <= alt:
                lo = mid
                h_lo = h_mid
            else:
                hi = mid
                h_hi = h_mid
        w = (alt - h_lo) / (h_hi - h_lo)
    u0 = _tri(u, it, iy, ix, lo, ft, fy, fx)
    u1 = _tri(u, it, iy, ix, lo + 1, ft, fy, fx)
    v0 = _tri(v, it, iy, ix, lo, ft, fy, fx)
    v1 = _tri(v, it, iy, ix, lo + 1, ft, fy, fx)
    return u0 + (u1 - u0) * w, v0 + (v1 - v0) * w, OK


@njit(cache=True)
def blended_wind(low, high, z_lo, z_hi, t, lat, lon, alt):
    """Wind from ``low`` below z_lo, ``high`` above z_hi, linear blend between.

    Used to put HRRR (top at 50 hPa) under GFS. If ``low`` is out of domain the
    ``high`` field is used. For a single source pass the same field twice with
    z_lo = z_hi = inf.
    """
    if alt <= z_lo:
        ul, vl, sl = wind_at(low, t, lat, lon, alt)
        if sl == OK:
            return ul, vl, OK
        return wind_at(high, t, lat, lon, alt)
    if alt >= z_hi:
        return wind_at(high, t, lat, lon, alt)
    uh, vh, sh = wind_at(high, t, lat, lon, alt)
    ul, vl, sl = wind_at(low, t, lat, lon, alt)
    if sl != OK:
        return uh, vh, sh
    if sh != OK:
        return ul, vl, sl
    w = (alt - z_lo) / (z_hi - z_lo)
    return ul + (uh - ul) * w, vl + (vh - vl) * w, OK
