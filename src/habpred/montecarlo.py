"""Monte Carlo launch uncertainty (plan step 4).

Each launch is flown many times with the balloon and forecast varied the way they
actually vary, then each landing is scored on the recoverability raster:

* **Ascent shape:** a real radiosonde climb profile (rate by altitude) picked at
  random and scaled to the planned mean rate. Balloons speed up with altitude by
  different amounts; one mean rate misses that (sonde test: worth about two days of
  forecast lead).
* **Mean ascent rate:** normal, ``ascent_sd`` fraction of the mean (fill error).
* **Burst altitude:** normal, ``burst_sd`` metres.
* **Descent rate:** uniform within +/- ``descent_frac`` of the sea-level rate.
* **Forecast error:** an error vector from the radiosonde hindcast at the nearest
  forecast lead (truth minus prediction with the sonde's own profile, so it holds
  only wind error), rotated into the predicted drift direction and added to the
  landing point.

The sonde-derived inputs (shapes and error vectors) are small and live in the
package (``mcdata/mc_inputs.npz``); ``scripts/build_mc_inputs.py`` rebuilds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .solver import predict_many

Z_GRID = np.arange(500.0, 40001.0, 1000.0)  # bin centres of profiles.rate_table
NORM_TOP = 30000.0  # shapes have a time-mean rate of 1 from the ground to here
LEAD_CLASSES = (6.0, 30.0, 54.0)  # hindcast lead classes (h before launch)


def default_inputs_path() -> Path:
    return Path(__file__).parent / "mcdata" / "mc_inputs.npz"


def normalise_shape(z, rate):
    """Rate table -> shape on Z_GRID with time-mean rate 1 over 0..NORM_TOP."""
    r = np.interp(Z_GRID, z, rate)
    below = Z_GRID < NORM_TOP
    mean = NORM_TOP / np.sum(1000.0 / r[below])
    return r / mean


def residual_frame(lat0, lon0, plat, plon, tlat, tlon):
    """Truth minus prediction, split along and across the predicted drift direction
    (cross positive to the left of travel). Returns (along_km, cross_km, pred_drift_km)."""
    kx = 111.32 * np.cos(np.radians(lat0))
    px, py = (plon - lon0) * kx, (plat - lat0) * 110.57
    ex, ey = (tlon - plon) * kx, (tlat - plat) * 110.57
    d = float(np.hypot(px, py))
    ux, uy = (px / d, py / d) if d > 0 else (1.0, 0.0)
    return float(ex * ux + ey * uy), float(-ex * uy + ey * ux), d


def apply_residual(lat0, lon0, plat, plon, along, cross):
    """Inverse of residual_frame, vectorised: shift predicted landings by (along, cross) km."""
    kx = 111.32 * np.cos(np.radians(lat0))
    px, py = (plon - lon0) * kx, (plat - lat0) * 110.57
    d = np.hypot(px, py)
    safe = np.where(d > 0, d, 1.0)
    ux, uy = np.where(d > 0, px / safe, 1.0), np.where(d > 0, py / safe, 0.0)
    ex = along * ux - cross * uy
    ey = along * uy + cross * ux
    return plat + ey / 110.57, plon + ex / kx


@dataclass
class Inputs:
    shapes: np.ndarray  # [S, len(Z_GRID)] ascent rate / mean rate
    lead: np.ndarray  # [R] lead class, h
    along: np.ndarray  # [R] km, truth minus prediction
    cross: np.ndarray  # [R] km
    drift: np.ndarray  # [R] predicted drift km (for reference)

    def save(self, path: Path | None = None):
        path = path or default_inputs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, z=Z_GRID, shapes=self.shapes.astype(np.float32),
                            lead=self.lead.astype(np.float32), along=self.along.astype(np.float32),
                            cross=self.cross.astype(np.float32), drift=self.drift.astype(np.float32))

    @classmethod
    def load(cls, path: Path | None = None) -> "Inputs":
        d = np.load(path or default_inputs_path())
        assert np.allclose(d["z"], Z_GRID)
        return cls(*(d[k].astype(np.float64) for k in ("shapes", "lead", "along", "cross", "drift")))

    def residuals(self, lead_h: float):
        cls = min(LEAD_CLASSES, key=lambda L: abs(L - lead_h))
        m = self.lead == cls
        return self.along[m], self.cross[m]


@dataclass
class Uncertainty:
    """Spread of the balloon and forecast. Means are the planned flight."""
    ascent: float  # mean ascent rate, m/s (time mean, ground to burst)
    burst: float  # m
    descent: float = 5.0  # sea-level descent rate, m/s
    ascent_sd: float = 0.05  # fraction of the mean
    burst_sd: float = 1500.0  # m
    descent_frac: float = 0.10  # uniform +/- fraction
    lead_h: float = 6.0  # forecast lead for the wind error
    shapes: bool = True  # sample real ascent shapes (else constant rate)
    wind_error: bool = True


@dataclass
class Members:
    asc_r: np.ndarray  # [n, K]
    burst: np.ndarray
    descent: np.ndarray
    along: np.ndarray
    cross: np.ndarray
    meta: dict = field(default_factory=dict)


def sample(unc: Uncertainty, n: int, rng: np.random.Generator, inputs: Inputs) -> Members:
    mean = unc.ascent * np.clip(1 + unc.ascent_sd * rng.standard_normal(n), 0.5, 1.5)
    if unc.shapes:
        shape = inputs.shapes[rng.integers(0, len(inputs.shapes), n)]
    else:
        shape = np.ones((n, Z_GRID.size))
    burst = unc.burst + unc.burst_sd * rng.standard_normal(n)
    descent = unc.descent * (1 + unc.descent_frac * rng.uniform(-1, 1, n))
    if unc.wind_error:
        a, c = inputs.residuals(unc.lead_h)
        k = rng.integers(0, len(a), n)
        along, cross = a[k], c[k]
    else:
        along = cross = np.zeros(n)
    return Members(mean[:, None] * shape, burst, descent, along, cross)


def landings(wind, t0: float, site_lat, site_lon, launch_alt: float, members: Members,
             ground_alt: float | None = None, upper=None, dt: float = 30.0):
    """Fly every member from every site (same members at each site, so sites compare
    like for like). Returns lat, lon, ok as [n_sites, n_members]; ok is False when
    the flight left the wind domain (far offshore)."""
    site_lat = np.atleast_1d(np.asarray(site_lat, dtype=np.float64))
    site_lon = np.atleast_1d(np.asarray(site_lon, dtype=np.float64))
    ns, nm = site_lat.size, members.burst.size
    lat0 = np.repeat(site_lat, nm)
    lon0 = np.repeat(site_lon, nm)
    tile = lambda x: np.tile(x, (ns,) + (1,) * (x.ndim - 1))  # noqa: E731
    ground = launch_alt if ground_alt is None else ground_alt
    r = predict_many(wind, t0, lat0, lon0, launch_alt, tile(members.asc_r), tile(members.burst),
                     tile(members.descent), ground, dt=dt, upper=upper, ascent_z=Z_GRID)
    lat, lon = apply_residual(lat0, lon0, r[:, 1], r[:, 2], tile(members.along),
                              tile(members.cross))
    ok = r[:, 4] == 0
    return lat.reshape(ns, nm), lon.reshape(ns, nm), ok.reshape(ns, nm)
