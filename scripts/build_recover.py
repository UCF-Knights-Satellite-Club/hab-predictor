"""Build the landing recoverability raster and write maps (plan step 3).

Needs data/osm/florida-latest.osm.pbf and data/nwi/FL_geodatabase_wetlands.zip
(see README); NLCD tiles are fetched automatically.

    uv run python scripts/build_recover.py [--force] [--out out/recover]

Writes data/masks/recover_*.npz (used by the predictor scripts) plus, in --out,
GeoTIFFs of tier/cover/walk time for QGIS or Google Earth and PNG maps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from habpred import recover as R
from habpred.solver import haversine_km

# places to zoom on: (name, lat, lon, half-size deg)
ZOOMS = [("KAOS-1 landing (Hal Scott)", *R.KAOS1_LANDING, 0.06),
         ("Orlando east / Econ", 28.55, -81.15, 0.25),
         ("Indian River Lagoon / Cape", 28.45, -80.70, 0.30)]

TIER_COLORS = {R.NO: "#9e9e9e", R.HARD: "#f4a259", R.OK: "#f2e863", R.EASY: "#4caf50"}
COVER_COLORS = {0: "#ffffff", R.WATER: "#3d7ab8", R.SWAMP: "#5b3f8c", R.RESTRICTED: "#d7263d",
                R.WET: "#7fb7a4", R.FOREST: "#2e6b3a", R.SHRUB: "#a3a86b", R.OPEN: "#e6d9a2",
                R.DEV_LOW: "#c9b2a6", R.DEV_HIGH: "#7a5c58"}


def write_tifs(rec, out):
    import rasterio
    from rasterio.transform import from_origin
    tr = from_origin(rec.lon_min, rec.lat_max, rec.res, rec.res)
    for name, arr, colors in (("tier", rec.tier, TIER_COLORS), ("cover", rec.cover, COVER_COLORS),
                              ("walk_dmin", rec.walk, None)):
        p = out / f"recover_{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype=arr.dtype, crs="EPSG:4326", transform=tr,
                           compress="deflate", tiled=True,
                           nodata=65535 if name == "walk_dmin" else None) as dst:
            dst.write(arr, 1)
            if colors:
                dst.write_colormap(1, {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) + (255,)
                                       for k, v in colors.items()})
        print(f"wrote {p}")


def plot(rec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    lat_min, lat_max, lon_min, lon_max = rec.bounds
    def cmap(colors):
        n = max(colors) + 1
        return ListedColormap([colors.get(i, "#ffffff") for i in range(n)]), n - 0.5

    tcm, tmax = cmap(TIER_COLORS)
    ccm, cmax = cmap(COVER_COLORS)
    # overview: decimate by 8 (mode-free; just sample) to keep the PNG light
    s = 8
    fig, axes = plt.subplots(1, 2, figsize=(18, 11))
    for ax, arr, cm, vmax, title, legend in (
            (axes[0], rec.tier, tcm, tmax, "Recoverability tier", TIER_COLORS),
            (axes[1], rec.cover, ccm, cmax, "Land cover class", COVER_COLORS)):
        ax.imshow(arr[::s, ::s], cmap=cm, vmin=-0.5, vmax=vmax, interpolation="nearest",
                  extent=(lon_min, lon_max, lat_min, lat_max))
        ax.set_aspect(1 / np.cos(np.radians(28.4)))
        ax.set_title(title)
        names = R.TIER_NAMES if legend is TIER_COLORS else R.COVER_NAMES
        ax.legend(handles=[Patch(color=c, label=names[k]) for k, c in legend.items() if k in names],
                  loc="lower left", fontsize=8)
    fig.suptitle(f"Florida landing recoverability ({rec.res * 111:.2f} km cells; tiers: easy <= "
                 f"{R.EASY_MIN:g} min, ok <= {R.OK_MIN:g} min, hard <= {R.HARD_MIN:g} min off trail)")
    fig.tight_layout()
    fig.savefig(out / "recover_overview.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out / 'recover_overview.png'}")

    fig, axes = plt.subplots(2, len(ZOOMS), figsize=(6 * len(ZOOMS), 11))
    for k, (name, lat, lon, h) in enumerate(ZOOMS):
        i0, j0, _ = rec.index(lat + h, lon - h)
        i1, j1, _ = rec.index(lat - h, lon + h)
        ext = (lon - h, lon + h, lat - h, lat + h)
        for row, (arr, cm, vmax) in enumerate(((rec.tier, tcm, tmax), (rec.cover, ccm, cmax))):
            ax = axes[row, k]
            ax.imshow(arr[i0:i1, j0:j1], cmap=cm, vmin=-0.5, vmax=vmax, interpolation="nearest",
                      extent=ext)
            rr, cc = np.nonzero(rec.road[i0:i1, j0:j1])
            if len(rr) < 200_000:
                ax.plot(lon - h + (cc + 0.5) * rec.res, lat + h - (rr + 0.5) * rec.res, ",k",
                        alpha=0.4)
            ax.plot(*R.KAOS1_LANDING[::-1], "r*", ms=12)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
            ax.set_aspect(1 / np.cos(np.radians(lat)))
            ax.set_title(f"{name}: {'tier' if row == 0 else 'cover'}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "recover_zooms.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out / 'recover_zooms.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="out/recover")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = R.build(force=a.force)

    land = (rec.cover != 0) & (rec.cover != R.WATER)
    print("\nLand area by cover:")
    for c, n in R.COVER_NAMES.items():
        if c in (0, R.WATER):
            continue
        print(f"  {n:32s} {np.mean(rec.cover[land] == c):6.1%}")
    print("Land area by tier:")
    for t, n in R.TIER_NAMES.items():
        print(f"  {n:6s} {np.mean(rec.tier[land] == t):6.1%}")

    lat, lon = R.KAOS1_LANDING
    i, j, _ = rec.index(lat, lon)
    rr, cc = np.nonzero(rec.road[i - 40:i + 41, j - 40:j + 41])
    d_road = (haversine_km(lat, lon, rec.lat_max - (i - 40 + rr + 0.5) * rec.res,
                           rec.lon_min + (j - 40 + cc + 0.5) * rec.res).min() * 1000
              if len(rr) else np.inf)
    print(f"\nKAOS-1 landing check: cover={R.COVER_NAMES[int(rec.cover[i, j])]}, "
          f"walk={rec.walk[i, j] / 10:.0f} min, tier={R.TIER_NAMES[int(rec.tier[i, j])]}, "
          f"nearest mapped road/trail {d_road:.0f} m (actual: ~300 m, 30 min)")
    write_tifs(rec, out)
    plot(rec, out)


if __name__ == "__main__":
    main()
