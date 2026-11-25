"""Coarse land/water mask from Natural Earth 1:10m land and lakes.

A first-pass stand-in for the recoverability raster (step 3): it only says
whether a point is on land, not whether that land is recoverable. Natural Earth
is public domain. Download the zips into ``data/ne/``:

    https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip
    https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip
    https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_north_america.zip
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapefile
import shapely
from shapely.geometry import box, shape

from .sources.common import DEFAULT_BBOX, data_dir


def _read_zip_shapes(zpath: Path, clip):
    with zipfile.ZipFile(zpath) as z:
        stem = next(n[:-4] for n in z.namelist() if n.endswith(".shp"))
        r = shapefile.Reader(shp=z.open(stem + ".shp"), shx=z.open(stem + ".shx"),
                             dbf=z.open(stem + ".dbf"))
        geoms = []
        for s in r.iterShapes():
            if s.bbox and not box(*s.bbox).intersects(clip):
                continue
            g = shape(s.__geo_interface__)
            if g.intersects(clip):
                geoms.append(g.intersection(clip))
        return geoms


@dataclass
class LandMask:
    lat0: float
    lon0: float
    res: float
    land: np.ndarray  # bool [lat, lon], row 0 = lat0

    def is_land(self, lat, lon):
        i = np.floor((np.asarray(lat) - self.lat0) / self.res).astype(int)
        j = np.floor((np.asarray(lon) - self.lon0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.land.shape[0]) & (j >= 0) & (j < self.land.shape[1])
        out = np.zeros(np.shape(i), dtype=bool)
        out[ok] = self.land[i[ok], j[ok]]
        return out

    @classmethod
    def build(cls, bbox=DEFAULT_BBOX, res: float = 0.005, force: bool = False) -> "LandMask":
        lat_min, lat_max, lon_min, lon_max = bbox
        tag = "_".join(f"{b:g}" for b in bbox)
        cache = data_dir() / "masks" / f"ne_land_{tag}_{res:g}.npz"
        if cache.exists() and not force:
            z = np.load(cache)
            return cls(float(z["lat0"]), float(z["lon0"]), float(z["res"]), z["land"])
        ne = data_dir() / "ne"
        clip = box(lon_min, lat_min, lon_max, lat_max)
        land = shapely.union_all(_read_zip_shapes(ne / "ne_10m_land.zip", clip))
        lakes = _read_zip_shapes(ne / "ne_10m_lakes.zip", clip) + \
            _read_zip_shapes(ne / "ne_10m_lakes_north_america.zip", clip)
        if lakes:
            land = land.difference(shapely.union_all(lakes))
        shapely.prepare(land)
        lats = lat_min + res * (np.arange(int(round((lat_max - lat_min) / res))) + 0.5)
        lons = lon_min + res * (np.arange(int(round((lon_max - lon_min) / res))) + 0.5)
        xx, yy = np.meshgrid(lons, lats)
        mask = shapely.contains_xy(land, xx, yy)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, lat0=lat_min, lon0=lon_min, res=res, land=mask)
        return cls(lat_min, lon_min, res, mask)
