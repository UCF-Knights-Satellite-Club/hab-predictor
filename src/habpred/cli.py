"""Command-line interface.

    habpred predict --lat 28.6134 --lon -81.1742 --time 2026-11-26T15:00Z \\
        --ascent 5 --burst 30000 --descent 5 [--alt 20] [--cycle latest] [--hrrr] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone

from .solver import haversine_km, predict


def _time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise argparse.ArgumentTypeError("time needs a zone, e.g. 2026-11-26T15:00Z")
    return t.astimezone(timezone.utc)


def cmd_predict(a):
    from .sources import gfs

    cycle = gfs.latest_cycle() if a.cycle == "latest" else _time(a.cycle)
    end = a.time + timedelta(hours=a.max_hours)
    step = 1 if (end - cycle).total_seconds() <= 120 * 3600 else 3
    wind = gfs.load(cycle, a.time, end, step_hours=step)
    upper = None
    if a.hrrr:
        from .sources import hrrr
        upper, wind = wind, hrrr.load(a.hrrr_cycle and _time(a.hrrr_cycle) or cycle, a.time, end)
    f = predict(wind, a.time, a.lat, a.lon, a.alt, a.ascent, a.burst, a.descent,
                ground_alt=a.ground, dt=a.dt, upper=upper, max_hours=a.max_hours)
    if f.status != 0:
        raise SystemExit(f"prediction failed (status {f.status}): left the wind grid or time range")
    bt, blat, blon, balt = f.burst
    t_land, lat, lon = f.landing
    fmt = "%Y-%m-%d %H:%M:%SZ"
    print(f"winds:   {wind.source}" + (f" + {upper.source} above 20 km" if upper else ""))
    print(f"burst:   {datetime.fromtimestamp(bt, timezone.utc):{fmt}}  {blat:.5f}, {blon:.5f}"
          f"  {balt:.0f} m")
    print(f"landing: {datetime.fromtimestamp(t_land, timezone.utc):{fmt}}  {lat:.5f}, {lon:.5f}"
          f"  ({haversine_km(a.lat, a.lon, lat, lon):.1f} km from launch, "
          f"{(t_land - f.t[0]) / 60:.0f} min)")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_utc", "lat", "lon", "alt_m"])
            for t, la, lo, z in zip(f.times_utc(), f.lat, f.lon, f.alt):
                w.writerow([t.strftime(fmt), f"{la:.6f}", f"{lo:.6f}", f"{z:.1f}"])
        print(f"wrote {a.csv}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="habpred")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="predict one flight from GFS (optionally HRRR below 20 km)")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True, help="degrees east (-180..180)")
    p.add_argument("--alt", type=float, default=0.0, help="launch altitude m")
    p.add_argument("--ground", type=float, default=None, help="landing ground altitude m")
    p.add_argument("--time", type=_time, required=True, help="launch time, ISO 8601 with zone")
    p.add_argument("--ascent", type=float, required=True, help="ascent rate m/s")
    p.add_argument("--burst", type=float, required=True, help="burst altitude m")
    p.add_argument("--descent", type=float, required=True, help="sea-level descent rate m/s")
    p.add_argument("--cycle", default="latest", help="GFS cycle (ISO time) or 'latest'")
    p.add_argument("--hrrr", action="store_true", help="use HRRR below 18-20 km")
    p.add_argument("--hrrr-cycle", default=None, help="HRRR cycle (default: same as GFS)")
    p.add_argument("--dt", type=float, default=30.0)
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--csv", help="write the trajectory to CSV")
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
