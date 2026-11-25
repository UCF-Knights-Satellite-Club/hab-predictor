"""Landing recoverability raster for Florida (plan step 3).

Every ~40 m cell gets a land cover class, a walking time from the nearest road
or trail, and a recoverability tier:

* ``NO``: open water, swamp wetter than the KAOS-1 recovery site, restricted
  ground (military, airports, prisons), or more than ``HARD_MIN`` minutes' walk;
* ``HARD``: up to ``HARD_MIN`` minutes off trail, or dense urban (roofs, wires);
* ``OK``: up to ``OK_MIN`` minutes, KAOS-1 with a little margin (it took 30 min to
  walk ~300 m; the raster puts it 341 m and 36 min from the nearest mapped trail);
* ``EASY``: up to ``EASY_MIN`` minutes.

Sources (all public, downloaded to ``data/``):

* NLCD 2021 land cover (MRLC WCS, public domain), 30 m;
* National Wetlands Inventory, Florida geodatabase (USFWS, public domain):
  the Cowardin water regime decides swamp vs passable wet ground;
* OpenStreetMap Florida extract (Geofabrik, ODbL): roads, tracks, paths and
  restricted areas.

Swamp rule, calibrated on KAOS-1: the landing point (28.4718, -81.1253) is NLCD
"woody wetlands" but outside every NWI polygon; the nearest NWI wetlands are a
semipermanently flooded cypress stand (PFO2F) and marsh (PEM1F). So palustrine
wetlands with a temporarily flooded or saturated regime (A, B, J) count as
passable wet ground, while seasonally or wetter regimes (C, E, F, G, H, K, ...)
and tidal wetlands count as swamp.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sources.common import data_dir, get

# --- grid -------------------------------------------------------------------

# Peninsula from Tampa Bay / Big Bend to the Atlantic, Lake Okeechobee to Georgia.
BBOX = (26.0, 30.8, -83.8, -79.9)  # lat_min, lat_max, lon_min, lon_max
RES = 0.0004  # degrees, ~44 m N-S x ~39 m E-W

# --- cover classes ----------------------------------------------------------

WATER, SWAMP, RESTRICTED, WET, FOREST, SHRUB, OPEN, DEV_LOW, DEV_HIGH = range(1, 10)
COVER_NAMES = {0: "none", WATER: "water", SWAMP: "swamp", RESTRICTED: "restricted",
               WET: "wet woods/marsh (KAOS-1 like)", FOREST: "forest", SHRUB: "shrub/scrub",
               OPEN: "open (pasture, crops, grass)", DEV_LOW: "developed (low/medium)",
               DEV_HIGH: "developed (high)"}

NLCD_TO_COVER = {0: WATER, 11: WATER, 12: WATER, 21: DEV_LOW, 22: DEV_LOW, 23: DEV_LOW,
                 24: DEV_HIGH, 31: OPEN, 41: FOREST, 42: FOREST, 43: FOREST, 52: SHRUB,
                 71: OPEN, 81: OPEN, 82: OPEN, 90: WET, 95: WET}

# Off-trail walking pace, minutes per km. WET is set from KAOS-1: 300 m in 30 min.
PACE_MIN_PER_KM = {WET: 100.0, FOREST: 40.0, SHRUB: 60.0, OPEN: 20.0, DEV_LOW: 15.0,
                   DEV_HIGH: 15.0}

# Tiers
NO, HARD, OK, EASY = 0, 1, 2, 3
TIER_NAMES = {NO: "no", HARD: "hard", OK: "ok", EASY: "easy"}
EASY_MIN, OK_MIN, HARD_MIN = 15.0, 40.0, 90.0

# NWI water regimes that are passable (temporarily flooded, saturated, intermittently flooded)
PASSABLE_REGIMES = set("ABJ")

KAOS1_LANDING = (28.47177, -81.12527)


# --- lookup -----------------------------------------------------------------

@dataclass
class Recoverability:
    lat_max: float
    lon_min: float
    res: float
    cover: np.ndarray  # uint8 [row, col], row 0 = north edge
    walk: np.ndarray   # uint16 walking time from road/trail, deci-minutes; 65535 = unreachable
    tier: np.ndarray   # uint8, from compute_tier (recomputed on load, so thresholds can change)
    road: np.ndarray   # bool, cell touches a road/track/path

    @property
    def shape(self):
        return self.cover.shape

    @property
    def bounds(self):
        rows, cols = self.shape
        return (self.lat_max - rows * self.res, self.lat_max, self.lon_min,
                self.lon_min + cols * self.res)

    def index(self, lat, lon):
        i = np.floor((self.lat_max - np.asarray(lat, float)) / self.res).astype(np.int64)
        j = np.floor((np.asarray(lon, float) - self.lon_min) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return i, j, ok

    def _at(self, arr, lat, lon, fill):
        i, j, ok = self.index(lat, lon)
        out = np.full(np.shape(i), fill, dtype=arr.dtype)
        out[ok] = arr[i[ok], j[ok]]
        return out

    def tier_at(self, lat, lon):
        """Tier at each point; outside the raster counts as NO."""
        return self._at(self.tier, lat, lon, NO)

    def cover_at(self, lat, lon):
        return self._at(self.cover, lat, lon, 0)

    def walk_min_at(self, lat, lon):
        w = self._at(self.walk, lat, lon, 65535).astype(float) / 10
        return np.where(w >= 6553.5, np.inf, w)

    def is_land(self, lat, lon):
        c = self.cover_at(lat, lon)
        return (c != 0) & (c != WATER)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, lat_max=self.lat_max, lon_min=self.lon_min, res=self.res,
                            cover=self.cover, walk=self.walk,
                            road=np.packbits(self.road, axis=1), cols=self.shape[1])
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | None = None) -> "Recoverability":
        z = np.load(path or default_path())
        road = np.unpackbits(z["road"], axis=1, count=int(z["cols"])).astype(bool)
        cover, walk = z["cover"], z["walk"]
        return cls(float(z["lat_max"]), float(z["lon_min"]), float(z["res"]), cover, walk,
                   compute_tier(cover, walk), road)


def compute_tier(cover, walk):
    """Tier from cover class and walking time (deci-minutes)."""
    tier = np.full(cover.shape, NO, np.uint8)
    passable = np.isin(cover, list(PACE_MIN_PER_KM))
    for t, limit in ((HARD, HARD_MIN), (OK, OK_MIN), (EASY, EASY_MIN)):
        tier[passable & (walk <= limit * 10)] = t
    tier[(cover == DEV_HIGH) & (tier > HARD)] = HARD
    return tier


def default_path(bbox=BBOX, res=RES) -> Path:
    tag = "_".join(f"{b:g}" for b in bbox)
    return data_dir() / "masks" / f"recover_{tag}_{res:g}.npz"


# --- build ------------------------------------------------------------------

def _grid(bbox, res):
    from rasterio.transform import from_origin
    lat_min, lat_max, lon_min, lon_max = bbox
    rows = int(round((lat_max - lat_min) / res))
    cols = int(round((lon_max - lon_min) / res))
    return from_origin(lon_min, lat_max, res, res), (rows, cols)


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


NLCD_WCS = ("https://www.mrlc.gov/geoserver/mrlc_download/wcs?service=WCS&version=2.0.1"
            "&request=GetCoverage&coverageId=mrlc_download__NLCD_2021_Land_Cover_L48"
            "&subsettingCrs=http://www.opengis.net/def/crs/EPSG/0/4326"
            "&subset=Lat({:.4f},{:.4f})&subset=Long({:.4f},{:.4f})&format=image/geotiff")


def nlcd(bbox=BBOX, res=RES, tile=0.5, margin=0.03, workers=4) -> np.ndarray:
    """NLCD 2021 classes on the target grid (0 = no data, e.g. offshore)."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    transform, shape = _grid(bbox, res)
    out = np.zeros(shape, np.uint8)
    cache = data_dir() / "nlcd"
    cache.mkdir(parents=True, exist_ok=True)
    lat_min, lat_max, lon_min, lon_max = bbox
    tiles = [(la, lo) for la in np.arange(lat_min, lat_max, tile)
             for lo in np.arange(lon_min, lon_max, tile)]

    def fetch(t):
        la, lo = t
        p = cache / f"nlcd2021_{la:.2f}_{lo:.2f}_{tile:g}.tif"
        if not p.exists():
            url = NLCD_WCS.format(la - margin, min(la + tile, lat_max) + margin,
                                  lo - margin, min(lo + tile, lon_max) + margin)
            data = get(url)
            if not data.startswith((b"II*", b"MM\0*")):
                raise RuntimeError(f"NLCD WCS returned non-TIFF for tile {t}: {data[:200]!r}")
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
        return p

    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(fetch, tiles))
    for p in paths:
        with rasterio.open(p) as src:
            b = src.bounds
            win = from_bounds(b.left, b.bottom, b.right, b.top, transform).round_offsets().round_lengths()
            r0, c0 = max(0, win.row_off), max(0, win.col_off)
            r1, c1 = min(shape[0], win.row_off + win.height), min(shape[1], win.col_off + win.width)
            if r1 <= r0 or c1 <= c0:
                continue
            sub = out[r0:r1, c0:c1].copy()
            reproject(src.read(1), sub, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform * transform.translation(c0, r0), dst_crs="EPSG:4326",
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
                      init_dest_nodata=False)
            out[r0:r1, c0:c1] = sub
    _log(f"NLCD: {len(paths)} tiles, {np.mean(out == 0):.1%} no data")
    return out


_REGIME_RE = re.compile(r"([A-Z])[a-z]*$")


def nwi_class(attr: str) -> int:
    """Map a Cowardin code (e.g. PFO2F, PEM1/SS1C, L1UBHh, E2EM1P) to a cover class, or 0."""
    if not attr:
        return 0
    system = attr[0]
    if system in "LM" or attr.startswith(("E1", "R1", "R2", "R3", "R5")):
        return WATER
    if attr.startswith(("E2", "M2")):
        return SWAMP  # tidal marsh, mangrove, flats
    if attr.startswith("R4"):
        return WET  # intermittent streambed
    if system == "P":
        if attr[1:3] in ("UB", "AB"):
            return WATER  # ponds
        m = _REGIME_RE.search(attr)
        regime = m.group(1) if m else ""
        return WET if regime in PASSABLE_REGIMES else SWAMP
    return 0


def _read_vector(path, layer, columns, where=None, bbox_ll=None):
    """Read features as (shapely geometries in lon/lat, {column: array})."""
    import pyogrio
    import pyogrio.raw
    import shapely
    from pyproj import CRS, Transformer

    info = pyogrio.read_info(path, layer=layer)
    crs = CRS.from_user_input(info["crs"]) if info["crs"] else CRS.from_epsg(4326)
    bbox = None
    tr = None
    if not crs.equals(CRS.from_epsg(4326)):
        tr = Transformer.from_crs(crs, 4326, always_xy=True)
        if bbox_ll is not None:
            back = Transformer.from_crs(4326, crs, always_xy=True)
            bbox = back.transform_bounds(*bbox_ll, densify_pts=21)
    else:
        bbox = bbox_ll
    _meta, _fids, wkb, fields = pyogrio.raw.read(path, layer=layer, columns=columns, where=where,
                                             bbox=bbox)
    geoms = shapely.from_wkb(wkb)
    if tr is not None:
        geoms = shapely.transform(geoms, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))
    return geoms, dict(zip(columns, fields))


def nwi(bbox=BBOX, res=RES, path: Path | None = None) -> np.ndarray:
    """NWI wetlands rasterized to cover classes (0 = no NWI polygon)."""
    import pyogrio
    from rasterio.features import rasterize

    transform, shape = _grid(bbox, res)
    zp = path or data_dir() / "nwi" / "FL_geodatabase_wetlands.zip"
    src = f"/vsizip/{zp}"
    inner = [n for n in _zip_names(zp) if n.rstrip("/").endswith(".gdb")]
    if inner:
        src += "/" + inner[0].rstrip("/")
    layer = next(name for name, _ in pyogrio.list_layers(src) if name.lower().endswith("wetlands"))
    lat_min, lat_max, lon_min, lon_max = bbox
    geoms, f = _read_vector(src, layer, ["ATTRIBUTE"], bbox_ll=(lon_min, lat_min, lon_max, lat_max))
    cls = np.array([nwi_class(a or "") for a in f["ATTRIBUTE"]], np.uint8)
    keep = cls > 0
    _log(f"NWI: {len(cls)} polygons in bbox, " + ", ".join(
        f"{COVER_NAMES[c]} {np.sum(cls == c)}" for c in (WATER, SWAMP, WET)))
    out = rasterize(zip(geoms[keep], cls[keep]), out_shape=shape, transform=transform, fill=0,
                    dtype=np.uint8)
    return out


def _zip_names(zp):
    import zipfile
    with zipfile.ZipFile(zp) as z:
        names = {n.split("/")[0] + "/" for n in z.namelist() if "/" in n}
    return sorted(names)


EXCLUDED_HIGHWAYS = {"motorway", "motorway_link", "proposed", "construction", "abandoned",
                     "disused", "platform", "raceway", "bus_guideway", "corridor", "elevator",
                     "razed", "no", "via_ferrata"}


def _tag(other_tags, key):
    if not other_tags:
        return None
    m = re.search(rf'"{re.escape(key)}"=>"([^"]*)"', other_tags)
    return m.group(1) if m else None


def osm(bbox=BBOX, res=RES, path: Path | None = None):
    """(road bool raster, restricted bool raster, restricted area names) from OpenStreetMap."""
    import os

    from rasterio.features import rasterize

    os.environ.setdefault("OSM_MAX_TMPFILE_SIZE", "4096")
    transform, shape = _grid(bbox, res)
    pbf = str(path or data_dir() / "osm" / "florida-latest.osm.pbf")
    lat_min, lat_max, lon_min, lon_max = bbox
    bb = (lon_min, lat_min, lon_max, lat_max)

    geoms, f = _read_vector(pbf, "lines", ["highway", "other_tags"],
                            where="highway IS NOT NULL", bbox_ll=bb)
    keep = np.array([h not in EXCLUDED_HIGHWAYS and _tag(o, "access") != "no"
                     for h, o in zip(f["highway"], f["other_tags"])])
    _log(f"OSM: {keep.sum()} of {len(keep)} highway lines usable for access")
    road = rasterize(((g, 1) for g in geoms[keep]), out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8, all_touched=True).astype(bool)

    geoms, f = _read_vector(
        pbf, "multipolygons", ["name", "landuse", "military", "aeroway", "amenity", "other_tags"],
        where="landuse = 'military' OR military IS NOT NULL OR aeroway = 'aerodrome' "
              "OR amenity = 'prison' OR name LIKE '%Launch Complex%'", bbox_ll=bb)
    names = sorted({n for n in f["name"] if n})
    _log(f"OSM: {len(geoms)} restricted polygons")
    restricted = rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                           fill=0, dtype=np.uint8).astype(bool)
    return road, restricted, names


def walk_minutes(cover, road, res=RES, lat_mid=28.4):
    """Least walking time (minutes) from any road cell over passable cover (inf if none)."""
    from skimage.graph import MCP_Geometric

    pace = np.full(256, np.inf)
    for c, p in PACE_MIN_PER_KM.items():
        pace[c] = p / 1000.0  # minutes per metre
    cost = pace[cover]
    dy = res * 111_320.0
    dx = dy * np.cos(np.radians(lat_mid))
    starts = np.argwhere(road & np.isfinite(cost))
    _log(f"walk: {len(starts)} start cells, {np.isfinite(cost).mean():.1%} passable")
    m = MCP_Geometric(cost, sampling=(dy, dx), fully_connected=True)
    cum, _ = m.find_costs(starts)
    cum[~np.isfinite(cost)] = np.inf
    return cum


def build(bbox=BBOX, res=RES, force=False) -> Recoverability:
    path = default_path(bbox, res)
    if path.exists() and not force:
        return Recoverability.load(path)
    t0 = time.time()
    lut = np.zeros(256, np.uint8)
    for k, v in NLCD_TO_COVER.items():
        lut[k] = v
    cover = lut[nlcd(bbox, res)]
    w = nwi(bbox, res)
    # NWI overrides NLCD: wet ground first, then swamp, then open water
    for c in (WET, SWAMP, WATER):
        cover[w == c] = c
    del w
    road, restricted, names = osm(bbox, res)
    cover[restricted] = RESTRICTED
    _log("restricted areas: " + "; ".join(names[:60]) + (" ..." if len(names) > 60 else ""))

    cum = walk_minutes(cover, road, res, lat_mid=0.5 * (bbox[0] + bbox[1]))
    walk = np.where(np.isfinite(cum), np.minimum(np.round(cum * 10), 65534), 65535).astype(np.uint16)
    del cum
    rec = Recoverability(bbox[1], bbox[2], res, cover, walk, compute_tier(cover, walk), road)
    rec.save(path)
    _log(f"built {path} in {time.time() - t0:.0f} s")
    return rec
