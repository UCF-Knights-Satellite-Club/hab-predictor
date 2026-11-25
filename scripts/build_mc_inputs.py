"""Build the Monte Carlo inputs from the radiosonde data (plan step 4).

* Ascent shapes: every usable sonde climb profile in data/sondes, as rate by
  altitude divided by its mean rate (src/habpred/montecarlo.py, normalise_shape).
* Forecast error vectors: truth minus prediction at the last fix heard, from
  scripts/sonde_hindcast.py's CSV (the ``asflown`` runs, so only wind error).

    uv run python scripts/build_mc_inputs.py [--hindcast out/sondes/sonde_hindcast.csv]

Writes src/habpred/mcdata/mc_inputs.npz (committed; small).
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from habpred import montecarlo as MC
from habpred.profiles import clean_track, flight_profile
from habpred.sources import sondes
from habpred.sources.common import data_dir


def shapes():
    summ = [json.loads(l) for l in (data_dir() / "sondes" / "summaries.jsonl").read_text().splitlines()]
    summ = list({s["serial"]: s for s in summ}.values())
    out, kept, seen = [], 0, 0
    for s in summ:
        if s["first_alt"] > 2500 or not sondes.track_path(s["serial"]).exists():
            continue
        seen += 1
        prof = flight_profile(*clean_track(*sondes.load_track(s["serial"])))
        if prof is None:
            continue
        tab = np.array(prof["asc_table"])
        if tab[:, 0].min() > 3000 or tab[:, 0].max() < 25000:
            continue
        shp = MC.normalise_shape(tab[:, 0], tab[:, 1])
        if not (0.4 < shp.min() and shp.max() < 2.5):
            continue
        out.append(shp)
        kept += 1
    print(f"ascent shapes: {kept} of {seen} sonde tracks usable")
    return np.array(out)


def residuals(path):
    lead, along, cross, drift = [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != "asflown" or not r.get("res_along_km"):
                continue
            lead.append(min(MC.LEAD_CLASSES, key=lambda L: abs(L - float(r["lead_h"]))))
            along.append(float(r["res_along_km"]))
            cross.append(float(r["res_cross_km"]))
            drift.append(float(r["pred_drift_km"]))
    lead, along, cross, drift = map(np.array, (lead, along, cross, drift))
    for L in MC.LEAD_CLASSES:
        m = lead == L
        e = np.hypot(along[m], cross[m])
        print(f"errors at {L:.0f} h lead: n={m.sum()}, along mean {along[m].mean():+.1f} km "
              f"(sd {along[m].std():.1f}), cross mean {cross[m].mean():+.1f} km "
              f"(sd {cross[m].std():.1f}), |e| median {np.median(e):.1f} km")
    return lead, along, cross, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", default="out/sondes/sonde_hindcast.csv")
    a = ap.parse_args()
    shp = shapes()
    q = np.percentile(shp, [10, 50, 90], axis=0)
    for z in (2500, 10500, 20500, 29500, 34500):
        k = int(np.searchsorted(MC.Z_GRID, z))
        print(f"  shape at {MC.Z_GRID[k] / 1000:4.1f} km: p10 {q[0, k]:.2f}  median {q[1, k]:.2f}"
              f"  p90 {q[2, k]:.2f}")
    inp = MC.Inputs(shp, *residuals(a.hindcast))
    inp.save()
    print(f"wrote {MC.default_inputs_path()}")


if __name__ == "__main__":
    main()
