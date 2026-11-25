"""February drift climatology for central Florida launch sites.

For every February day of the chosen years, fly a grid of launch sites with GFS
12Z near-analysis winds (f000-f008, fetched by step2_fetch.py stage ``feb``) and
record where each flight lands. Reports, per launch site and flight profile:

* P(land): fraction of days the landing point is on land (Natural Earth mask);
* P(rec): lands somewhere recoverable (tier hard or better in the step 3 raster,
  scripts/build_recover.py), and P(ok): tier ok or easy, i.e. no worse than KAOS-1;
* median and 90th-percentile drift distance.

    uv run python scripts/feb_drift.py [--years 2024 2025 2026] [--out out/feb]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from habpred import recover as R
from habpred.landmask import LandMask
from habpred.solver import haversine_km, predict_many
from habpred.sites import COUNTY, GRID, LAUNCH_HOURS_UTC, NAMED, PROFILES, load_day

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--out", default="out/feb")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mask = LandMask.build()
    rec = R.Recoverability.load()
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
        tier = np.where(ok, rec.tier_at(lat, lon), R.NO)
        i, j, inside = rec.index(lat, lon)
        outside_land = land & ~inside
        drift = np.where(ok, haversine_km(site_lat[None], site_lon[None], lat, lon), np.inf)
        summary[p] = dict(p_land=land.mean(axis=0), p_rec=(tier >= R.HARD).mean(axis=0),
                          p_ok=(tier >= R.OK).mean(axis=0), med=np.median(drift, axis=0),
                          p90=np.percentile(drift, 90, axis=0), lat=lat, lon=lon, land=land)
        print(f"{p}: {outside_land.mean():.1%} of landings are on land outside the "
              f"recoverability raster (counted as not recoverable)")

    # named sites table
    rows = []
    print(f"\n{'site':24s}" + "".join(f"{'|':>3s} {p:6s}{'land':>6s}{'rec':>5s}{'ok':>5s}"
                                        f"{'med':>6s}{'p90':>5s}" for p in PROFILES))
    for k, n in enumerate(names):
        i = n_grid + k
        line = f"{n[:24]:24s}"
        row = {"site": n, "county": COUNTY[n], "lat": site_lat[i], "lon": site_lon[i]}
        for p in PROFILES:
            s = summary[p]
            line += (f"{'|':>3s} {'':6s}{s['p_land'][i]:6.0%}{s['p_rec'][i]:5.0%}{s['p_ok'][i]:5.0%}"
                     f"{s['med'][i]:6.0f}{s['p90'][i]:5.0f}")
            row.update({f"{p}_p_land": round(float(s["p_land"][i]), 3),
                        f"{p}_p_rec": round(float(s["p_rec"][i]), 3),
                        f"{p}_p_ok": round(float(s["p_ok"][i]), 3),
                        f"{p}_median_km": round(float(s["med"][i]), 1),
                        f"{p}_p90_km": round(float(s["p90"][i]), 1)})
        rows.append(row)
        print(line)
    print("(land: any land; rec: tier hard+; ok: tier ok/easy, no worse than KAOS-1; "
          "med/p90: drift km)")
    with open(out / "named_sites.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # grid CSV
    with open(out / "grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon", "launch_on_land"] + [f"{p}_{m}" for p in PROFILES
                                                       for m in ("p_land", "p_rec", "p_ok",
                                                                 "median_km", "p90_km")])
        on_land = mask.is_land(site_lat[:n_grid], site_lon[:n_grid])
        for i in range(n_grid):
            w.writerow([f"{site_lat[i]:.3f}", f"{site_lon[i]:.3f}", int(on_land[i])] +
                       [f"{summary[p][m][i]:.3f}" for p in PROFILES
                        for m in ("p_land", "p_rec", "p_ok", "med", "p90")])
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
        fig, axes2 = plt.subplots(2, 3, figsize=(20, 13))
        axes = axes2.ravel()
        for ax, key, cm, label, title in (
                (axes[0], "p_land", "RdYlGn", "P(lands on land)", "Chance of a land landing"),
                (axes[1], "p_rec", "RdYlGn", "P(tier hard or better)",
                 "Chance of a recoverable landing (hard+)"),
                (axes[2], "p_ok", "RdYlGn", "P(tier ok or easy)",
                 "Chance of an ok/easy landing (<= KAOS-1 effort)"),
                (axes[3], "med", "viridis", "median drift km", "Median drift distance"),
                (axes[4], "p90", "viridis", "p90 drift km", "90th percentile drift distance")):
            z = np.where(launch_land, s[key][:n_grid].reshape(shape), np.nan)
            kw = dict(vmin=0, vmax=1) if key.startswith("p_") else {}
            im = ax.pcolormesh(glon, glat, z, cmap=cm, shading="nearest", **kw)
            fig.colorbar(im, ax=ax, shrink=0.8, label=label)
            ax.set_title(title)
        # 6: landing scatter for named sites, one colour per county
        ax = axes[5]
        counties = list(dict.fromkeys(COUNTY[n] for n in names))
        color = {c: plt.cm.tab10(j % 10) for j, c in enumerate(counties)}
        for c in counties:
            idx = [n_grid + k for k, n in enumerate(names) if COUNTY[n] == c]
            members = [n for n in names if COUNTY[n] == c]
            label = f"{c}: {', '.join(members)}" if len(members) <= 2 else \
                f"{c}: {len(members)} sites"
            ax.scatter(s["lon"][:, idx].ravel(), s["lat"][:, idx].ravel(), s=4, alpha=0.5,
                       label=label, color=color[c])
        for k, n in enumerate(names):
            ax.plot(site_lon[n_grid + k], site_lat[n_grid + k], "^", ms=7,
                    color=color[COUNTY[n]], mec="k", mew=0.6, zorder=5)
        ax.set_title("Landing points from named sites by launch county (all days)")
        ax.legend(fontsize=6, markerscale=3, loc="lower left")
        for ax in axes:
            ax.contour(mlon, mlat, mask.land.astype(float), levels=[0.5], colors="k",
                       linewidths=0.6)
            if ax is not axes[5]:
                for k, n in enumerate(names):
                    ax.plot(site_lon[n_grid + k], site_lat[n_grid + k], "k^", ms=4)
            ax.set_aspect(1 / np.cos(np.radians(28.5)))
            ax.set_xlim(-83.6, -78.5) if ax is axes[5] else ax.set_xlim(glon[0] - .3, glon[-1] + .3)
            ax.set_ylim(26.5, 31.0) if ax is axes[5] else ax.set_ylim(glat[0] - .3, glat[-1] + .3)
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
