"""Compare our predictor with SondeHub's Tawhiri on the same GFS cycle.

Makes one SondeHub API request per case (4 by default), then runs our model on
the GFS cycle SondeHub reports using, in turn, Tawhiri-like settings (0.5 deg,
3-hourly, 60 s steps) and our default (0.25 deg, hourly, 30 s steps).

    uv run python scripts/compare_sondehub.py [--hours-ahead 24]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np

from habpred import sondehub
from habpred.solver import haversine_km, predict
from habpred.sources import gfs

SITES = {
    "Econ River WA": (28.6134, -81.1742, 20.0),
    "Brooksville": (28.55, -82.39, 30.0),
    "Lake Wales": (27.90, -81.59, 40.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours-ahead", type=float, default=24.0)
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    base = (now + timedelta(hours=args.hours_ahead)).replace(minute=0, second=0, microsecond=0)
    cases = [(name, base, *SITES[name], 5.0, 30000.0, 5.0) for name in SITES]
    cases.append(("Econ River WA +48h fast", base + timedelta(hours=48), *SITES["Econ River WA"],
                  7.5, 33000.0, 5.0))

    print(f"{'case':26s} {'model':28s} {'landing':>20s} {'vs SH km':>8s} {'dur min':>7s}")
    for name, t, lat, lon, alt, ar, burst, dr in cases:
        sh, req = sondehub.predict(t, lat, lon, alt, ar, burst, dr)
        cycle = datetime.fromisoformat(req["dataset"].replace("Z", "+00:00"))
        end = datetime.fromtimestamp(sh.t[-1], timezone.utc) + timedelta(minutes=30)
        wf = gfs.load(cycle, t, end, step_hours=1 if (end - cycle).total_seconds() <= 120 * 3600
                      else 3)
        print(f"{name:26s} {'SondeHub (GFS ' + cycle.strftime('%m-%d %HZ') + ')':28s} "
              f"{sh.lat[-1]:9.4f},{sh.lon[-1]:9.4f} {'':>8s} {(sh.t[-1] - sh.t[0]) / 60:7.1f}")
        variants = [("ours 0.5/3h dt60", wf.subsample(2, 3) if len(wf.times) > 3 and
                     wf.times[1] - wf.times[0] == 3600 else wf.subsample(2, 1), 60.0),
                    ("ours 0.25/1h dt30", wf, 30.0)]
        for label, field, dt in variants:
            f = predict(field, t, lat, lon, alt, ar, burst, dr, ground_alt=sh.alt[-1], dt=dt)
            d = haversine_km(sh.lat[-1], sh.lon[-1], f.lat[-1], f.lon[-1])
            print(f"{'':26s} {label:28s} {f.lat[-1]:9.4f},{f.lon[-1]:9.4f} {d:8.2f} "
                  f"{(f.t[-1] - f.t[0]) / 60:7.1f}")
        # burst point comparison
        bi = sh.burst_index
        fb = predict(wf, t, lat, lon, alt, ar, burst, dr, ground_alt=sh.alt[-1], dt=30.0)
        db = haversine_km(sh.lat[bi], sh.lon[bi], fb.burst[1], fb.burst[2])
        drift = haversine_km(lat, lon, sh.lat[-1], sh.lon[-1])
        print(f"{'':26s} burst-point diff {db:.2f} km; SondeHub total drift {drift:.0f} km\n")


if __name__ == "__main__":
    main()
