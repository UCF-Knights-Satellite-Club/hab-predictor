"""Re-run Florida radiosonde flights with GFS forecasts and score the predictions.

Each flight starts from its first received fix (time, position, altitude), so no
launch time guess is needed. Two flight-profile variants:

* ``asflown``: ascent and descent rates binned by altitude from the sonde's own
  track, and its real burst altitude. The remaining error is the wind forecast.
* ``constant``: the flight's mean ascent rate, real burst altitude and a sea-level
  descent rate fitted to its descent (density-scaled), i.e. what a Tawhiri-style
  predictor would do with perfect knowledge of the balloon. The gap between
  the two shows how much profile shape matters.

Errors reported per flight and lead class:
* ``burst_km``: predicted vs real burst position;
* ``end_km``: predicted vs real position at the last fix heard on descent (matched
  by altitude), and ``drift_km`` from the first fix to that last fix.

    uv run python scripts/sonde_hindcast.py [--months 2026-02 ...] [--offline] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from habpred.montecarlo import residual_frame
from habpred.profiles import clean_track, flight_profile
from habpred.solver import haversine_km, predict
from habpred.sources import gfs, sondes
from habpred.sources.common import data_dir
from habpred.wind import WindField

LEAD_CLASSES_H = (3, 27, 51)


def load_cached(cycle, start, end, offline):
    f0 = max(0, int((start - cycle).total_seconds() // 3600))
    f1 = -int(-(end - cycle).total_seconds() // 3600)
    if f1 > 120:
        return None
    fields = []
    for fh in range(f0, f1 + 1):
        p = gfs._cache_path(cycle, fh, gfs.DEFAULT_BBOX)
        if p.exists():
            fields.append(WindField.load(p))
        elif offline:
            return None
        else:
            fields.append(gfs.download_hour(cycle, fh))
    return WindField.concat_time(fields)


def position_at_alt(f, z):
    """Predicted (lat, lon) where the descent first passes altitude z."""
    i0 = f.burst_index
    a = f.alt[i0:]
    k = np.argmax(a <= z)
    if a[k] > z:
        return f.lat[-1], f.lon[-1]
    k += i0
    w = (f.alt[k - 1] - z) / max(1e-9, f.alt[k - 1] - f.alt[k])
    return f.lat[k - 1] + w * (f.lat[k] - f.lat[k - 1]), f.lon[k - 1] + w * (f.lon[k] - f.lon[k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2025-12", "2026-01", "2026-02", "2026-03"])
    ap.add_argument("--offline", action="store_true", help="skip flights whose GFS isn't cached")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/sondes")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    months = {tuple(int(x) for x in m.split("-")) for m in a.months}

    summ = [json.loads(l) for l in (data_dir() / "sondes" / "summaries.jsonl").read_text().splitlines()]
    summ = list({s["serial"]: s for s in summ}.values())
    summ.sort(key=lambda s: s["first_time"])
    rows = []
    n_used = 0
    for s in summ:
        t_first = datetime.fromisoformat(s["first_time"].replace("Z", "+00:00"))
        if (t_first.year, t_first.month) not in months or s["first_alt"] > 2500:
            continue
        if not sondes.track_path(s["serial"]).exists():
            continue
        prof = flight_profile(*clean_track(*sondes.load_track(s["serial"])))
        if prof is None:
            continue
        launch = datetime.fromtimestamp(prof["t0"], timezone.utc)
        end_t = prof.get("end_t", prof["burst_t"]) + 600
        end = datetime.fromtimestamp(end_t, timezone.utc)
        any_run = False
        for lead in LEAD_CLASSES_H:
            c = launch - timedelta(hours=lead)
            cycle = c.replace(minute=0, second=0, microsecond=0) - timedelta(hours=c.hour % 6)
            wf = load_cached(cycle, launch - timedelta(minutes=30), end, a.offline)
            if wf is None:
                continue
            variants = {"asflown": (prof["asc_table"], prof.get("des_table") or 5.0)}
            variants["constant"] = (prof["asc_mean"], prof.get("des_v0", 5.0))
            for vname, (asc, des) in variants.items():
                # stop at the last heard altitude (or right after burst if no descent heard)
                stop_alt = prof.get("end_alt", prof["burst_alt"] - 1.0)
                f = predict(wf, launch, prof["lat0"], prof["lon0"], prof["alt0"], asc,
                            prof["burst_alt"], des, ground_alt=stop_alt, dt=30.0)
                if f.status != 0 or f.burst_index < 0:
                    continue
                any_run = True
                row = {
                    "serial": s["serial"], "site": s["site"], "launch": launch.isoformat(),
                    "month": launch.month, "cycle": cycle.strftime("%Y%m%d%H"),
                    "lead_h": round((launch - cycle).total_seconds() / 3600, 1),
                    "variant": vname, "burst_alt": round(prof["burst_alt"]),
                    "burst_drift_km": round(float(haversine_km(prof["lat0"], prof["lon0"],
                                                               prof["burst_lat"], prof["burst_lon"])), 1),
                    "burst_km": round(float(haversine_km(prof["burst_lat"], prof["burst_lon"],
                                                         f.burst[1], f.burst[2])), 2),
                }
                if "end_t" in prof:
                    plat, plon = position_at_alt(f, prof["end_alt"])
                    along, cross, pdrift = residual_frame(prof["lat0"], prof["lon0"], plat, plon,
                                                          prof["end_lat"], prof["end_lon"])
                    row.update(pred_drift_km=round(pdrift, 1), res_along_km=round(along, 2),
                               res_cross_km=round(cross, 2))
                    row.update(end_alt=round(prof["end_alt"]),
                               drift_km=round(float(haversine_km(prof["lat0"], prof["lon0"],
                                                                 prof["end_lat"], prof["end_lon"])), 1),
                               end_km=round(float(haversine_km(prof["end_lat"], prof["end_lon"],
                                                               plat, plon)), 2))
                rows.append(row)
        n_used += any_run
        if a.limit and n_used >= a.limit:
            break

    if not rows:
        raise SystemExit("no flights could be scored (missing tracks or GFS data)")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                  if k in rows[0] else 99)
    with open(out / "sonde_hindcast.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"{n_used} flights scored, {len(rows)} runs -> {out / 'sonde_hindcast.csv'}\n")
    groups = defaultdict(list)
    for r in rows:
        lead_class = min(LEAD_CLASSES_H, key=lambda L: abs(r["lead_h"] - (L + 3)))
        groups[(r["variant"], lead_class)].append(r)
    print(f"{'variant':9s} {'lead':>6s} {'n':>4s} {'burst err km':>18s} {'end err km':>18s} "
          f"{'drift km':>9s} {'end err % drift':>16s}")
    for (v, L), rs in sorted(groups.items()):
        b = np.array([r["burst_km"] for r in rs])
        e = np.array([r["end_km"] for r in rs if "end_km" in r])
        d = np.array([r["drift_km"] for r in rs if "end_km" in r])
        pct = 100 * e / np.maximum(d, 1)
        fmt = lambda x: f"{np.median(x):6.1f} / {np.percentile(x, 90):6.1f}" if len(x) else "-"
        print(f"{v:9s} {L + 3:4d} h {len(rs):4d} {fmt(b):>18s} {fmt(e):>18s} "
              f"{np.median(d) if len(d) else 0:9.0f} {fmt(pct):>16s}")
    print("\n(median / 90th percentile)")


if __name__ == "__main__":
    main()
