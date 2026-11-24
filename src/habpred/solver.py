"""Balloon flight integration: constant-rate ascent to burst, then parachute descent.

The model matches Tawhiri's ``standard_profile`` (the predictor behind SondeHub):

* horizontal velocity equals the interpolated wind;
* ascent is a constant ``ascent_rate`` until ``burst_alt``;
* descent speed is ``descent_rate * sqrt(rho0 / rho(alt))``;
* fixed-step RK4, with the burst and landing instants found by bisection.

Ground is a constant altitude for now (a DEM can replace it later).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from numba import njit, prange

from .atmosphere import descent_rate
from .wind import OK, OUT_OF_DOMAIN, WindField, blended_wind

EARTH_RADIUS = 6371009.0

ASCENT = 0
DESCENT = 1

# Solver status codes (in addition to wind OK / OUT_OF_DOMAIN).
LANDED = 0
MAX_STEPS = 2


@njit(cache=True)
def _deriv(low, high, z_lo, z_hi, t, lat, lon, alt, phase, rates):
    """``rates`` = (asc_z, asc_r, descent_v0, desc_z, desc_r).

    Ascent rate is interpolated from the (asc_z, asc_r) table (one row = constant).
    Descent uses the (desc_z, desc_r) table if it is non-empty, else the drag model
    scaled from the sea-level rate ``descent_v0``.
    """
    asc_z, asc_r, descent_v0, desc_z, desc_r = rates
    u, v, st = blended_wind(low, high, z_lo, z_hi, t, lat, lon, alt)
    r = EARTH_RADIUS + alt
    dlat = math.degrees(v / r)
    dlon = math.degrees(u / (r * math.cos(math.radians(lat))))
    if phase == ASCENT:
        dalt = np.interp(alt, asc_z, asc_r)
    elif desc_z.shape[0] > 0:
        dalt = -np.interp(alt, desc_z, desc_r)
    else:
        dalt = -descent_rate(alt, descent_v0)
    return dlat, dlon, dalt, st


@njit(cache=True)
def _rk4(low, high, z_lo, z_hi, t, lat, lon, alt, h, phase, rates):
    k1a, k1o, k1z, s1 = _deriv(low, high, z_lo, z_hi, t, lat, lon, alt, phase, rates)
    k2a, k2o, k2z, s2 = _deriv(low, high, z_lo, z_hi, t + h / 2, lat + h / 2 * k1a,
                               lon + h / 2 * k1o, alt + h / 2 * k1z, phase, rates)
    k3a, k3o, k3z, s3 = _deriv(low, high, z_lo, z_hi, t + h / 2, lat + h / 2 * k2a,
                               lon + h / 2 * k2o, alt + h / 2 * k2z, phase, rates)
    k4a, k4o, k4z, s4 = _deriv(low, high, z_lo, z_hi, t + h, lat + h * k3a,
                               lon + h * k3o, alt + h * k3z, phase, rates)
    st = s1 | s2 | s3 | s4
    return (lat + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a),
            lon + h / 6 * (k1o + 2 * k2o + 2 * k3o + k4o),
            alt + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z), st)


@njit(cache=True)
def _done(phase, alt, burst_alt, ground_alt):
    if phase == ASCENT:
        return alt >= burst_alt
    return alt <= ground_alt


@njit(cache=True)
def integrate(low, high, z_lo, z_hi, t0, lat0, lon0, alt0, rates, burst_alt,
              ground_alt, dt, out):
    """Fly one balloon. ``out`` is a preallocated [n, 4] array (t, lat, lon, alt).

    Returns (n_points, burst_index, status). The last row is the landing point.
    """
    t, lat, lon, alt = t0, lat0, lon0, alt0
    n_max = out.shape[0]
    out[0, 0] = t
    out[0, 1] = lat
    out[0, 2] = lon
    out[0, 3] = alt
    n = 1
    burst_idx = -1
    phase = ASCENT
    if alt >= burst_alt:
        phase = DESCENT
        burst_idx = 0
    while True:
        if n >= n_max:
            return n, burst_idx, MAX_STEPS
        nlat, nlon, nalt, st = _rk4(low, high, z_lo, z_hi, t, lat, lon, alt, dt, phase, rates)
        if st != OK:
            return n, burst_idx, OUT_OF_DOMAIN
        h = dt
        if _done(phase, nalt, burst_alt, ground_alt):
            # bisect on the step length to find the crossing time
            lo, hi = 0.0, dt
            while hi - lo > 0.01:
                mid = 0.5 * (lo + hi)
                mlat, mlon, malt, _ = _rk4(low, high, z_lo, z_hi, t, lat, lon, alt, mid,
                                           phase, rates)
                if _done(phase, malt, burst_alt, ground_alt):
                    hi = mid
                else:
                    lo = mid
            h = hi
            nlat, nlon, nalt, st = _rk4(low, high, z_lo, z_hi, t, lat, lon, alt, h, phase, rates)
            nalt = burst_alt if phase == ASCENT else ground_alt
        t += h
        lat, lon, alt = nlat, nlon, nalt
        out[n, 0] = t
        out[n, 1] = lat
        out[n, 2] = lon
        out[n, 3] = alt
        n += 1
        if _done(phase, alt, burst_alt, ground_alt):
            if phase == ASCENT:
                phase = DESCENT
                burst_idx = n - 1
            else:
                return n, burst_idx, LANDED


@njit(parallel=True, cache=True)
def landings(low, high, z_lo, z_hi, t0, lat0, lon0, alt0, ascent_rate, burst_alt,
             descent_v0, ground_alt, dt, n_buf):
    """Run many flights in parallel. All parameter arguments are 1-D arrays of equal
    length. Returns an [N, 5] array of (t, lat, lon, burst_t, status)."""
    n_runs = t0.shape[0]
    res = np.empty((n_runs, 5))
    for i in prange(n_runs):
        buf = np.empty((n_buf, 4))
        asc_z = np.zeros(1)
        asc_r = np.full(1, ascent_rate[i])
        rates = (asc_z, asc_r, descent_v0[i], np.empty(0), np.empty(0))
        n, b, st = integrate(low, high, z_lo, z_hi, t0[i], lat0[i], lon0[i], alt0[i],
                             rates, burst_alt[i], ground_alt[i], dt, buf)
        res[i, 0] = buf[n - 1, 0]
        res[i, 1] = buf[n - 1, 1]
        res[i, 2] = buf[n - 1, 2]
        res[i, 3] = buf[b, 0] if b >= 0 else np.nan
        res[i, 4] = st
    return res


# ----- Python-friendly wrapper ------------------------------------------

@dataclass
class Flight:
    t: np.ndarray  # unix seconds
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    burst_index: int
    status: int

    @property
    def landing(self) -> tuple[float, float, float]:
        return float(self.t[-1]), float(self.lat[-1]), float(self.lon[-1])

    @property
    def burst(self) -> tuple[float, float, float, float]:
        i = self.burst_index
        return float(self.t[i]), float(self.lat[i]), float(self.lon[i]), float(self.alt[i])

    def times_utc(self) -> list[datetime]:
        return [datetime.fromtimestamp(x, timezone.utc) for x in self.t]


def _sources(wind: WindField, upper: WindField | None, blend: tuple[float, float]):
    if upper is None:
        f = wind.as_tuple()
        return f, f, math.inf, math.inf
    return wind.as_tuple(), upper.as_tuple(), float(blend[0]), float(blend[1])


def _profile(rate) -> tuple[np.ndarray, np.ndarray]:
    """A scalar rate, or a table of (altitude m, rate m/s) rows sorted by altitude."""
    a = np.asarray(rate, dtype=np.float64)
    if a.ndim == 0:
        return np.zeros(1), a.reshape(1)
    a = a[np.argsort(a[:, 0])]
    return np.ascontiguousarray(a[:, 0]), np.ascontiguousarray(a[:, 1])


def predict(wind: WindField, launch_time: datetime, lat: float, lon: float, launch_alt: float,
            ascent_rate, burst_alt: float, descent_rate,
            ground_alt: float | None = None, dt: float = 30.0,
            upper: WindField | None = None, blend: tuple[float, float] = (18000.0, 20000.0),
            max_hours: float = 12.0) -> Flight:
    """Predict one flight.

    ``wind`` supplies winds (and, with ``upper`` set, only the low levels, e.g. HRRR);
    ``upper`` supplies winds above ``blend[1]`` with a linear blend from ``blend[0]``.
    ``ascent_rate`` is m/s, or a table of (altitude, rate) rows interpolated linearly.
    ``descent_rate`` is the sea-level descent rate in m/s (scaled with air density), or a
    table of (altitude, rate) rows used as-is. ``lon`` is degrees east in -180..180.
    """
    low, high, z_lo, z_hi = _sources(wind, upper, blend)
    ground = launch_alt if ground_alt is None else ground_alt
    asc_z, asc_r = _profile(ascent_rate)
    if np.ndim(descent_rate) == 0:
        rates = (asc_z, asc_r, float(descent_rate), np.empty(0), np.empty(0))
    else:
        dz, dr = _profile(descent_rate)
        rates = (asc_z, asc_r, 0.0, dz, dr)
    out = np.empty((int(max_hours * 3600 / dt) + 10, 4))
    n, b, st = integrate(low, high, z_lo, z_hi, launch_time.timestamp(), lat, lon, launch_alt,
                         rates, burst_alt, ground, dt, out)
    o = out[:n]
    return Flight(o[:, 0].copy(), o[:, 1].copy(), o[:, 2].copy(), o[:, 3].copy(), b, st)


def predict_many(wind: WindField, t0, lat, lon, launch_alt, ascent_rate, burst_alt,
                 descent_rate, ground_alt, dt: float = 30.0, upper: WindField | None = None,
                 blend: tuple[float, float] = (18000.0, 20000.0), max_hours: float = 12.0):
    """Vectorised landings. Scalars are broadcast; returns [N, 5] (t, lat, lon, burst_t, status)."""
    arrs = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in
                                 (t0, lat, lon, launch_alt, ascent_rate, burst_alt,
                                  descent_rate, ground_alt)])
    arrs = [np.ascontiguousarray(a.ravel()) for a in arrs]
    low, high, z_lo, z_hi = _sources(wind, upper, blend)
    return landings(low, high, z_lo, z_hi, *arrs, dt, int(max_hours * 3600 / dt) + 10)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))
