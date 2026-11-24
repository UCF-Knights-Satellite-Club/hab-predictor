"""Hindcast KAOS-1 (2025-08-30) with GFS forecasts at several lead times.

Usage:
    uv run python scripts/kaos1_hindcast.py SPOT_CSV [--hrrr] [--plot out/kaos1.png]

SPOT_CSV is KAOS-1_SPOTData.ods exported to CSV (LibreOffice:
``soffice --headless --convert-to csv KAOS-1_SPOTData.ods``). Columns: local time
(EDT), blank, "lat, lon", speed mph, blank, raw alt ft, rollover offset ft,
corrected alt ft.

Flight parameters come from the flight itself, so the remaining error is mostly
the wind forecast. Sources: wiki post-flight results and the SPOT rollover
analysis (onboard pressure logger, 5 s samples from 10:25:15 EDT):
* liftoff 10:07:00 EDT (SPOT shows the payload moving at 10:07:21);
* logger at 23,100 ft (7,041 m) at 10:25:15, steady climb to burst at 111,333 ft
  (33,935 m) at 11:23:30, i.e. ~7.69 m/s; the first 7 km therefore averaged ~6.41 m/s,
  which agrees with the SPOT fixes at 10:12 and 10:17;
* touchdown 12:10:21 (first logger sample within ~100 ft of ground pressure);
* measured fall rates by altitude band (rollover analysis table).

Two parameter sets are run for each forecast:
* "measured": the ascent and descent rate profiles above;
* "constant": what a predictor like SondeHub takes: one mean ascent rate and a
  sea-level descent rate tuned to the measured descent time.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone

import numpy as np

from habpred.atmosphere import descent_rate
from habpred.solver import haversine_km, predict
from habpred.sources import gfs

EDT = timezone(timedelta(hours=-4))
FT = 0.3048

LAUNCH_LAT, LAUNCH_LON, LAUNCH_ALT = 28.6134, -81.1742, 20.0
BURST_ALT = 111333 * FT
BURST_TIME = datetime(2025, 8, 30, 11, 23, 30, tzinfo=EDT)
GROUND_ALT = 20.0


def load_spot(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.reader(fh):
            if not r or not r[0].strip():
                continue
            t = datetime.strptime(r[0].strip(), "%m/%d/%Y %H:%M:%S").replace(tzinfo=EDT)
            lat, lon = (float(x) for x in r[2].split(","))
            alt = float(r[7]) * FT if len(r) > 7 and r[7].strip() else np.nan
            rows.append((t, lat, lon, alt))
    return rows


LAUNCH_TIME = datetime(2025, 8, 30, 10, 7, 0, tzinfo=EDT)
TOUCHDOWN = datetime(2025, 8, 30, 12, 10, 21, tzinfo=EDT)
LOGGER_START = (datetime(2025, 8, 30, 10, 25, 15, tzinfo=EDT), 23100 * FT)

# ascent: two segments split at the logger start
_r_low = (LOGGER_START[1] - LAUNCH_ALT) / (LOGGER_START[0] - LAUNCH_TIME).total_seconds()
_r_high = (BURST_ALT - LOGGER_START[1]) / (BURST_TIME - LOGGER_START[0]).total_seconds()
ASCENT_MEASURED = [[0.0, _r_low], [LOGGER_START[1], _r_low],
                   [LOGGER_START[1] + 1.0, _r_high], [40000.0, _r_high]]
ASCENT_CONSTANT = (BURST_ALT - LAUNCH_ALT) / (BURST_TIME - LAUNCH_TIME).total_seconds()

# descent: (band top ft, band bottom ft, fall rate ft/min) from the logger
_BANDS = [(111000, 100000, 9040), (100000, 80000, 7850), (80000, 60000, 4790),
          (60000, 40000, 2950), (40000, 25000, 1660), (25000, 10000, 1150),
          (10000, 2000, 980), (2000, 0, 1030)]
DESCENT_MEASURED = [[(a + b) / 2 * FT, r * FT / 60] for a, b, r in _BANDS]


def fit_constant_descent():
    """Sea-level rate whose drag-model descent time matches the measured one."""
    target = (TOUCHDOWN - BURST_TIME).total_seconds()
    z = np.linspace(GROUND_ALT, BURST_ALT, 4000)

    def model(v0):
        return float(np.trapezoid([1 / descent_rate(zz, v0) for zz in z], z))

    lo, hi = 2.0, 10.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if model(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def track_error_km(flight, spot):
    """Median distance between SPOT fixes and the predicted position at the same time."""
    t = np.array([s[0].timestamp() for s in spot])
    m = (t >= flight.t[0]) & (t <= flight.t[-1])
    plat = np.interp(t[m], flight.t, flight.lat)
    plon = np.interp(t[m], flight.t, flight.lon)
    lat = np.array([s[1] for s in spot])[m]
    lon = np.array([s[2] for s in spot])[m]
    return float(np.median(haversine_km(lat, lon, plat, plon)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spot_csv")
    ap.add_argument("--hrrr", action="store_true", help="also run HRRR below 50 hPa + GFS above")
    ap.add_argument("--plot", help="write a PNG map of the tracks")
    args = ap.parse_args()

    spot = load_spot(args.spot_csv)
    land_fix = next((la, lo) for t, la, lo, a in spot
                    if t > BURST_TIME + timedelta(minutes=30) and np.isnan(a))
    v0 = fit_constant_descent()
    launch = LAUNCH_TIME
    print(f"Launch    {LAUNCH_TIME:%H:%M:%S} EDT  ascent {_r_low:.2f} m/s to "
          f"{LOGGER_START[1]:.0f} m, then {_r_high:.2f} m/s (mean {ASCENT_CONSTANT:.2f})")
    print(f"Burst     {BURST_TIME:%H:%M:%S} EDT  {BURST_ALT:.0f} m")
    print(f"Touchdown {TOUCHDOWN:%H:%M:%S} EDT  constant-model sea-level descent {v0:.2f} m/s")
    print(f"Landing   {land_fix[0]:.5f}, {land_fix[1]:.5f} (SPOT)\n")
    land = land_fix
    params = {"measured": (ASCENT_MEASURED, DESCENT_MEASURED),
              "constant": (ASCENT_CONSTANT, v0)}

    end = TOUCHDOWN + timedelta(minutes=30)
    cycles = [datetime(2025, 8, 30, 12, tzinfo=timezone.utc) - timedelta(hours=6 * k)
              for k in range(0, 14)]
    runs = []
    print(f"{'run':44s} {'lead h':>6s} {'landing':>20s} {'err km':>7s} {'track km':>8s}")

    def run(label, lead, field, pset, dt, **kw):
        asc, desc = params[pset]
        f = predict(field, launch, LAUNCH_LAT, LAUNCH_LON, LAUNCH_ALT, asc, BURST_ALT, desc,
                    GROUND_ALT, dt=dt, **kw)
        if f.status != 0:
            print(f"{label}: status {f.status}")
            return
        err = haversine_km(land[0], land[1], f.lat[-1], f.lon[-1])
        dur = (f.t[-1] - f.t[0]) / 60
        print(f"{label:44s} {lead:6.1f} {f.lat[-1]:9.4f},{f.lon[-1]:9.4f} {err:7.1f} "
              f"{track_error_km(f, spot):8.1f}   ({dur:.0f} min)")
        runs.append((label, lead, f, err))

    for cyc in cycles:
        wf = gfs.load(cyc, launch, end)
        lead = (launch - cyc).total_seconds() / 3600
        tag = f"{cyc:%m-%d %HZ}"
        run(f"GFS 0.25/1h measured {tag}", lead, wf, "measured", 30.0)
        run(f"GFS 0.25/1h constant {tag}", lead, wf, "constant", 30.0)
        if len(wf.times) > 3:
            run(f"GFS 0.5/3h constant (Tawhiri-like) {tag}", lead, wf.subsample(2, 3),
                "constant", 60.0)

    if args.hrrr:
        from habpred.sources import hrrr
        print()
        for cyc in [datetime(2025, 8, 30, h, tzinfo=timezone.utc) for h in (12, 6, 0)] + \
                   [datetime(2025, 8, 29, 18, tzinfo=timezone.utc)]:
            low = hrrr.load(cyc, launch, end)
            # GFS from the latest cycle at or before the HRRR cycle
            gcyc = cyc - timedelta(hours=cyc.hour % 6)
            high = gfs.load(gcyc, launch, end)
            lead = (launch - cyc).total_seconds() / 3600
            run(f"HRRR+GFS measured {cyc:%m-%d %HZ}", lead, low, "measured", 30.0,
                upper=high, blend=(18000.0, 20000.0))

    if args.plot:
        plot(args.plot, spot, runs, land)


def plot(path, spot, runs, land):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    lat = [s[1] for s in spot]
    lon = [s[2] for s in spot]
    ax.plot(lon, lat, "k.-", lw=2, label="KAOS-1 SPOT track", zorder=5)
    ax.plot(LAUNCH_LON, LAUNCH_LAT, "g^", ms=12, label="Launch", zorder=6)
    ax.plot(land[1], land[0], "kX", ms=12, label="Landing", zorder=6)
    shown = [r for r in runs if "measured" in r[0]]
    cmap = plt.get_cmap("viridis")
    leads = [r[1] for r in shown] or [0]
    for label, lead, f, err in shown:
        c = cmap(lead / max(leads))
        ls = "--" if label.startswith("HRRR") else "-"
        ax.plot(f.lon, f.lat, ls, color=c, lw=1, alpha=0.8,
                label=f"{label} ({lead:.0f} h lead, {err:.1f} km)")
        ax.plot(f.lon[-1], f.lat[-1], "o", color=c, ms=4)
    ax.set_aspect(1 / np.cos(np.radians(28.55)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("KAOS-1 hindcast, 2025-08-30 (as-flown ascent, burst and descent)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
