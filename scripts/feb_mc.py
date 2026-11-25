"""February launch-site Monte Carlo (plan step 4.1).

Every named site, every February launch (as scripts/feb_drift.py), flown
--members times with the balloon and forecast varied (habpred.montecarlo). For each
launch this gives P(ok): the chance the landing is no harder to recover than KAOS-1.
A site's February is then split into launches that are a confident go (P(ok) >= 80%),
marginal (50-80%) and no-go (< 50%).

Also reports what a single deterministic forecast is worth: when it says "ok", how
often is the landing really ok? And, with --ablate, which uncertainty dominates the
footprint size (balloon or forecast).

    uv run python scripts/feb_mc.py [--years 2024 2025 2026] [--members 200]
        [--footprint-site MOSI] [--ablate] [--out out/feb_mc]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from habpred import montecarlo as MC
from habpred import recover as R
from habpred.solver import haversine_km, predict_many
from habpred.sites import COUNTY, LAUNCH_HOURS_UTC, NAMED, PROFILES, load_day

GO, MARGINAL = 0.8, 0.5
LAUNCH_ALT = 20.0


def spread_km(lat, lon):
    """80th-percentile distance of members from their median point, per row."""
    mlat, mlon = np.median(lat, axis=-1, keepdims=True), np.median(lon, axis=-1, keepdims=True)
    return np.percentile(haversine_km(mlat, mlon, lat, lon), 80, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--members", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--footprint-site", default="MOSI")
    ap.add_argument("--ablate", action="store_true",
                    help="also run kaos1 with only the balloon spread and only the wind error")
    ap.add_argument("--out", default="out/feb_mc")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = R.Recoverability.load()
    inputs = MC.Inputs.load()
    rng = np.random.default_rng(a.seed)
    names = list(NAMED)
    slat = np.array([NAMED[n][0] for n in names])
    slon = np.array([NAMED[n][1] for n in names])
    fp = names.index(a.footprint_site)

    variants = {p: MC.Uncertainty(ascent=ar, burst=b, descent=dr) for p, (ar, b, dr) in PROFILES.items()}
    if a.ablate:
        ar, b, dr = PROFILES["kaos1"]
        variants["kaos1 balloon only"] = MC.Uncertainty(ar, b, dr, wind_error=False)
        variants["kaos1 wind only"] = MC.Uncertainty(ar, b, dr, ascent_sd=0, burst_sd=0,
                                                     descent_frac=0, shapes=False)

    days = []
    for y in a.years:
        d = datetime(y, 2, 1, tzinfo=timezone.utc)
        while d.month == 2:
            days.append(d)
            d += timedelta(days=1)

    rows = []
    det_tier = {p: [] for p in PROFILES}
    res = {v: dict(p_ok=[], p_rec=[], spread=[]) for v in variants}
    foot = {p: [] for p in PROFILES}  # (launch label, det lat/lon, member lat/lon, p_ok)
    used = 0
    for day in days:
        wf = load_day(day)
        if wf is None:
            continue
        used += 1
        for hour in LAUNCH_HOURS_UTC:
            t0 = day.replace(hour=hour).timestamp()
            label = day.replace(hour=hour).strftime("%Y-%m-%d %H")
            for v, unc in variants.items():
                mem = MC.sample(unc, a.members, rng, inputs)
                lat, lon, ok = MC.landings(wf, t0, slat, slon, LAUNCH_ALT, mem)
                tier = np.where(ok, rec.tier_at(lat, lon), R.NO)
                p_ok = (tier >= R.OK).mean(axis=1)
                p_rec = (tier >= R.HARD).mean(axis=1)
                spr = spread_km(lat, lon)
                res[v]["p_ok"].append(p_ok)
                res[v]["p_rec"].append(p_rec)
                res[v]["spread"].append(spr)
                if v in PROFILES:
                    ar, b, dr = PROFILES[v]
                    r = predict_many(wf, t0, slat, slon, LAUNCH_ALT, ar, b, dr, LAUNCH_ALT)
                    dt_ = np.where(r[:, 4] == 0, rec.tier_at(r[:, 1], r[:, 2]), R.NO)
                    det_tier[v].append(dt_)
                    foot[v].append((label, r[fp, 1], r[fp, 2], lat[fp], lon[fp], p_ok[fp]))
                    for k, n in enumerate(names):
                        rows.append({"site": n, "county": COUNTY[n], "launch_utc": label,
                                     "profile": v, "det_tier": R.TIER_NAMES[int(dt_[k])],
                                     "p_ok": round(float(p_ok[k]), 3),
                                     "p_rec": round(float(p_rec[k]), 3),
                                     "spread80_km": round(float(spr[k]), 1)})
    if not used:
        raise SystemExit("no February GFS data found")
    n_launch = used * len(LAUNCH_HOURS_UTC)
    print(f"{used} February days x {len(LAUNCH_HOURS_UTC)} launch times = {n_launch} launches, "
          f"{a.members} members each, {len(inputs.shapes)} ascent shapes, "
          f"{len(inputs.residuals(6)[0])} forecast-error vectors (6 h lead)\n")

    with open(out / "launches.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for p in PROFILES:
        pk = np.array(res[p]["p_ok"])  # [launches, sites]
        dt_ = np.array(det_tier[p])
        det_ok = dt_ >= R.OK
        print(f"profile {p}: {'site':24s} {'det ok':>7s} {'MC ok':>6s} {'go':>5s} {'marg':>5s} "
              f"{'no-go':>6s} {'go/Feb':>7s} {'P(ok|det ok)':>13s} {'P(ok|det not)':>14s} "
              f"{'spread':>7s}")
        for k, n in enumerate(names):
            x = pk[:, k]
            go, marg = (x >= GO).mean(), ((x >= MARGINAL) & (x < GO)).mean()
            cond_ok = x[det_ok[:, k]].mean() if det_ok[:, k].any() else np.nan
            cond_no = x[~det_ok[:, k]].mean() if (~det_ok[:, k]).any() else np.nan
            spr = np.median(np.array(res[p]["spread"])[:, k])
            print(f"{'':{len(p) + 10}s}{n[:24]:24s} {det_ok[:, k].mean():7.0%} {x.mean():6.0%} "
                  f"{go:5.0%} {marg:5.0%} {1 - go - marg:6.0%} {go * 28.25:7.1f} "
                  f"{cond_ok:13.0%} {cond_no:14.0%} {spr:6.0f}k")
            summary.append({"site": n, "county": COUNTY[n], "profile": p,
                            "det_p_ok": round(float(det_ok[:, k].mean()), 3),
                            "mc_p_ok": round(float(x.mean()), 3), "go": round(float(go), 3),
                            "marginal": round(float(marg), 3),
                            "no_go": round(float(1 - go - marg), 3),
                            "go_days_per_feb": round(float(go * 28.25), 1),
                            "p_ok_given_det_ok": round(float(cond_ok), 3),
                            "p_ok_given_det_not_ok": round(float(cond_no), 3),
                            "median_spread80_km": round(float(spr), 1)})
        print()
    print(f"go: P(ok) >= {GO:.0%}; marginal: {MARGINAL:.0%}-{GO:.0%}; go/Feb: go launches per "
          f"February (one launch time); spread: median 80% radius of the landing cloud\n")
    with open(out / "sites.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    if a.ablate:
        print("Footprint size by uncertainty source (kaos1, median 80% radius over all sites "
              "and launches):")
        for v in variants:
            if v.startswith("kaos1"):
                s = np.array(res[v]["spread"])
                pk = np.array(res[v]["p_ok"])
                print(f"  {v:20s} {np.median(s):5.1f} km   mean P(ok) {pk.mean():.0%}")
        print()

    plot(out, names, res, foot, rec, a.footprint_site, n_launch, a.members)


def plot(out, names, res, foot, rec, fp_site, n_launch, members):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from build_recover import TIER_COLORS  # scripts/ is on sys.path when run as a script

    # 1: go / marginal / no-go per site
    profs = list(PROFILES)
    fig, axes = plt.subplots(1, len(profs), figsize=(8 * len(profs), 7), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), profs):
        pk = np.array(res[p]["p_ok"])
        go = (pk >= GO).mean(0)
        marg = ((pk >= MARGINAL) & (pk < GO)).mean(0)
        order = np.argsort(go + marg / 2)
        y = np.arange(len(names))
        ax.barh(y, go[order], color="#2e7d32", label=f"go (P(ok) >= {GO:.0%})")
        ax.barh(y, marg[order], left=go[order], color="#f2c14e",
                label=f"marginal ({MARGINAL:.0%}-{GO:.0%})")
        ax.barh(y, 1 - go[order] - marg[order], left=(go + marg)[order], color="#bdbdbd",
                label=f"no-go (< {MARGINAL:.0%})")
        ax.set_yticks(y, [names[i] for i in order], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of February launches")
        ar, b, dr = PROFILES[p]
        ax.set_title(f"{p}: {ar} m/s, burst {b / 1000:.1f} km")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"February launches by chance of an ok/easy landing ({n_launch} launches, "
                 f"{members} Monte Carlo members each)")
    fig.tight_layout()
    fig.savefig(out / "feb_mc_sites.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'feb_mc_sites.png'}")

    # 2: footprints for one site, six launches spanning the P(ok) range
    cm = ListedColormap([TIER_COLORS[t] for t in (R.NO, R.HARD, R.OK, R.EASY)])
    f = foot["kaos1"]
    pks = np.array([x[5] for x in f])
    order = np.argsort(pks)
    picks = order[np.linspace(0, len(order) - 1, 6).round().astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, i in zip(axes.ravel(), picks):
        label, dlat, dlon, mlat, mlon, p = f[i]
        # frame the central 96% of members; a few tail members may fall outside
        span = max(np.subtract(*np.percentile(mlat, [98, 2])), np.subtract(*np.percentile(mlon, [98, 2])))
        h = max(0.15, 0.65 * span)
        clat, clon = np.median(mlat), np.median(mlon)
        i0, j0, _ = rec.index(clat + h, clon - h)
        i1, j1, _ = rec.index(clat - h, clon + h)
        s = max(1, (i1 - i0) // 800)
        ax.imshow(rec.tier[i0:i1:s, j0:j1:s], cmap=cm, vmin=-0.5, vmax=3.5,
                  interpolation="nearest", extent=(clon - h, clon + h, clat - h, clat + h))
        ax.scatter(mlon, mlat, s=6, c="k", alpha=0.5, lw=0)
        ax.plot(dlon, dlat, "r*", ms=14, mec="w")
        ax.set_xlim(clon - h, clon + h)
        ax.set_ylim(clat - h, clat + h)
        ax.set_aspect(1 / np.cos(np.radians(clat)))
        ax.set_title(f"{label}Z launch: P(ok) = {p:.0%}", fontsize=10)
    fig.legend(handles=[Patch(color=TIER_COLORS[t], label=R.TIER_NAMES[t])
                        for t in (R.EASY, R.OK, R.HARD, R.NO)]
               + [plt.Line2D([], [], color="k", marker=".", ls="", label="Monte Carlo landing"),
                  plt.Line2D([], [], color="r", marker="*", ls="", ms=10,
                             label="single deterministic forecast")],
               loc="lower center", ncol=6)
    fig.suptitle(f"Landing footprints from {fp_site}, kaos1 profile: six February launches "
                 f"from lowest to highest P(ok)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "feb_mc_footprints.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out / 'feb_mc_footprints.png'}")


if __name__ == "__main__":
    main()
