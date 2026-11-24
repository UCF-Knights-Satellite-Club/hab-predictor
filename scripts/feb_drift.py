"""February drift climatology for central Florida launch sites.

For every February day of the chosen years, fly a grid of launch sites with GFS
12Z near-analysis winds (f000-f008, fetched by step2_fetch.py stage ``feb``) and
record where each flight lands. Reports, per launch site and flight profile:

* P(land): fraction of days the landing point is on land (Natural Earth mask, so
  the Indian River Lagoon counts as land; step 3 refines this);
* median and 90th-percentile drift distance.

    uv run python scripts/feb_drift.py [--years 2024 2025 2026] [--out out/feb]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from habpred.landmask import LandMask
from habpred.solver import haversine_km, predict_many
from habpred.sources import gfs
from habpred.wind import WindField

PROFILES = {
    # name: (ascent m/s, burst m, sea-level descent m/s)
    "kaos1": (7.4, 33900.0, 5.0),
    "slow": (5.0, 30000.0, 5.0),
}
LAUNCH_HOURS_UTC = (14, 16)  # 9 and 11 am EST
GRID = dict(lat=(27.3, 29.7), lon=(-82.9, -80.5), step=0.05)
NAMED = {
    "Econ River WA (KAOS-1)": (28.6134, -81.1742),
    "Clermont": (28.55, -81.77),
    "Bushnell": (28.66, -82.11),
    "Brooksville": (28.55, -82.39),
    "Inverness": (28.84, -82.33),
    "Ocala": (29.19, -82.14),
    "Dade City": (28.36, -82.20),
    "Lakeland": (28.04, -81.95),
    "Lake Wales": (27.90, -81.59),
    "Sebring": (27.50, -81.44),
    "Plant City": (28.02, -82.12),
}


def load_day(day: datetime) -> WindField | None:
    cyc = day.replace(hour=12)
    paths = [gfs._cache_path(cyc, fh, gfs.DEFAULT_BBOX) for fh in range(0, 9)]
    if not all(p.exists() for p in paths):
        return None
    return WindField.concat_time([WindField.load(p) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--out", default="out/feb")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mask = LandMask.build()
    glat = np.arange(GRID["lat"][0], GRID["lat"][1] + 1e-9, GRID["step"])
    glon = np.arange(GRID["lon"][0], GRID["lon"][1] + 1e-9, GRID["step"])
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    names = list(NAMED)
    site_lat = np.concatenate([LA.ravel(), [NAMED[n][0] for n in names]])
    site_lon = np.concatenate([LO.ravel(), [NAMED[n][1] for n in names]])
    n_grid = LA.size

    days = []
    for y in a.years:
        d = datetime(y, 2, 1, tzinfo=timezone.utc)
        while d.month == 2:
            days.append(d)
            d += timedelta(days=1)

    # results[profile] -> arrays [n_cases, n_sites] of landing lat/lon/status
    res = {p: {"lat": [], "lon": [], "ok": []} for p in PROFILES}
    used = 0
    for day in days:
        wf = load_day(day)
        if wf is None:
            continue
        used += 1
        for hour in LAUNCH_HOURS_UTC:
            t0 = day.replace(hour=hour).timestamp()
            for p, (ar, burst, dr) in PROFILES.items():
                r = predict_many(wf, t0, site_lat, site_lon, 20.0, ar, burst, dr, 20.0, dt=30.0)
                res[p]["lat"].append(r[:, 1])
                res[p]["lon"].append(r[:, 2])
                res[p]["ok"].append(r[:, 4] == 0)
    if used == 0:
        raise SystemExit("no February GFS data found; run step2_fetch.py --stages feb first")
    print(f"{used} February days x {len(LAUNCH_HOURS_UTC)} launch times, "
          f"{len(site_lat)} sites, profiles: {', '.join(PROFILES)}\n")

    summary = {}
    for p in PROFILES:
        lat = np.array(res[p]["lat"])
        lon = np.array(res[p]["lon"])
        ok = np.array(res[p]["ok"])
        land = ok & mask.is_land(lat, lon)  # out-of-domain = far offshore
        drift = np.where(ok, haversine_km(site_lat[None], site_lon[None], lat, lon), np.inf)
        summary[p] = dict(p_land=land.mean(axis=0), med=np.median(drift, axis=0),
                          p90=np.percentile(drift, 90, axis=0), lat=lat, lon=lon, land=land)

    # named sites table
    rows = []
    print(f"{'site':26s}" + "".join(f"{p + ' P(land)':>15s}{p + ' med km':>13s}{'p90':>6s}"
                                    for p in PROFILES))
    for k, n in enumerate(names):
        i = n_grid + k
        line = f"{n:26s}"
        row = {"site": n, "lat": site_lat[i], "lon": site_lon[i]}
        for p in PROFILES:
            s = summary[p]
            line += f"{s['p_land'][i]:15.0%}{s['med'][i]:13.0f}{s['p90'][i]:6.0f}"
            row.update({f"{p}_p_land": round(float(s["p_land"][i]), 3),
                        f"{p}_median_km": round(float(s["med"][i]), 1),
                        f"{p}_p90_km": round(float(s["p90"][i]), 1)})
        rows.append(row)
        print(line)
    with open(out / "named_sites.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # grid CSV
    with open(out / "grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon", "launch_on_land"] + [f"{p}_{m}" for p in PROFILES
                                                       for m in ("p_land", "median_km", "p90_km")])
        on_land = mask.is_land(site_lat[:n_grid], site_lon[:n_grid])
        for i in range(n_grid):
            w.writerow([f"{site_lat[i]:.3f}", f"{site_lon[i]:.3f}", int(on_land[i])] +
                       [f"{summary[p][m][i]:.3f}" for p in PROFILES
                        for m in ("p_land", "med", "p90")])
    plot(out, mask, glat, glon, n_grid, summary, site_lat, site_lon, names, used)


def plot(out, mask, glat, glon, n_grid, summary, site_lat, site_lon, names, used):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = (len(glat), len(glon))
    mlat = mask.lat0 + mask.res * (np.arange(mask.land.shape[0]) + 0.5)
    mlon = mask.lon0 + mask.res * (np.arange(mask.land.shape[1]) + 0.5)
    launch_land = mask.is_land(site_lat[:n_grid], site_lon[:n_grid]).reshape(shape)
    for p, s in summary.items():
        fig, axes = plt.subplots(1, 3, figsize=(20, 7.2))
        # 1: P(land) by launch site
        ax = axes[0]
        z = np.where(launch_land, s["p_land"][:n_grid].reshape(shape), np.nan)
        im = ax.pcolormesh(glon, glat, z, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="P(lands on land)")
        ax.set_title("Chance of a land landing, by launch site")
        # 2: median drift
        ax = axes[1]
        z = np.where(launch_land, s["med"][:n_grid].reshape(shape), np.nan)
        im = ax.pcolormesh(glon, glat, z, cmap="viridis", shading="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="median drift km")
        ax.set_title("Median drift distance")
        # 3: landing scatter for named sites
        ax = axes[2]
        for k, n in enumerate(names):
            i = n_grid + k
            ax.scatter(s["lon"][:, i], s["lat"][:, i], s=4, alpha=0.5, label=n)
        ax.set_title("Landing points from named sites (all days)")
        ax.legend(fontsize=6, markerscale=3, loc="lower left")
        for ax in axes:
            ax.contour(mlon, mlat, mask.land.astype(float), levels=[0.5], colors="k",
                       linewidths=0.6)
            for k, n in enumerate(names):
                ax.plot(site_lon[n_grid + k], site_lat[n_grid + k], "k^", ms=4)
            ax.set_aspect(1 / np.cos(np.radians(28.5)))
            ax.set_xlim(-83.6, -78.5) if ax is axes[2] else ax.set_xlim(glon[0] - .3, glon[-1] + .3)
            ax.set_ylim(26.5, 31.0) if ax is axes[2] else ax.set_ylim(glat[0] - .3, glat[-1] + .3)
        ar, burst, dr = PROFILES[p]
        fig.suptitle(f"February drift, profile '{p}' (ascent {ar} m/s, burst {burst / 1000:.1f} km, "
                     f"descent {dr} m/s), {used} days x {len(LAUNCH_HOURS_UTC)} launch times, "
                     f"GFS 12Z near-analysis")
        fig.tight_layout()
        fig.savefig(out / f"feb_drift_{p}.png", dpi=130)
        plt.close(fig)
        print(f"wrote {out / f'feb_drift_{p}.png'}")


if __name__ == "__main__":
    main()
