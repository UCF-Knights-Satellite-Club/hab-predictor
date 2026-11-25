"""Named launch sites, flight profiles and launch times shared by the climatology
scripts, plus loading of the cached daily 12Z GFS runs (step2_fetch.py ``feb`` and
``daily`` stages)."""

from __future__ import annotations

from datetime import datetime

from .sources import gfs
from .wind import WindField

PROFILES = {
    # name: (ascent m/s, burst m, sea-level descent m/s)
    "kaos1": (7.4, 33900.0, 5.0),
    "slow": (5.0, 30000.0, 5.0),
}
LAUNCH_HOURS_UTC = (14, 16)  # 9 and 11 am EST
GRID = dict(lat=(27.3, 29.7), lon=(-82.9, -80.5), step=0.05)
NAMED = {
    "Econ River WA (KAOS-1)": (28.6134, -81.1742),
    "Clermont": (28.55, -81.77),
    "Bushnell": (28.66, -82.11),
    "Brooksville": (28.55, -82.39),
    "Inverness": (28.84, -82.33),
    "Ocala": (29.19, -82.14),
    "Dade City": (28.36, -82.20),
    "Lakeland": (28.04, -81.95),
    "Lake Wales": (27.90, -81.59),
    "Sebring": (27.50, -81.44),
    "Plant City": (28.02, -82.12),
    # Tampa outreach sites (OSM centroids)
    "MOSI": (28.0537, -82.4046),
    "USF Tampa": (28.0618, -82.4138),
    "Adams Middle": (28.0424, -82.4658),
    "Middleton High": (27.9907, -82.4320),
    "Ruskin NWS (72210)": (27.7054, -82.4007),  # Tampa Bay radiosonde site
}
COUNTY = {
    "Econ River WA (KAOS-1)": "Orange", "Clermont": "Lake", "Bushnell": "Sumter",
    "Brooksville": "Hernando", "Inverness": "Citrus", "Ocala": "Marion", "Dade City": "Pasco",
    "Lakeland": "Polk", "Lake Wales": "Polk", "Sebring": "Highlands",
    "Plant City": "Hillsborough", "MOSI": "Hillsborough", "USF Tampa": "Hillsborough",
    "Adams Middle": "Hillsborough", "Middleton High": "Hillsborough",
    "Ruskin NWS (72210)": "Hillsborough",
}


def load_day(day: datetime) -> WindField | None:
    """12Z GFS f000-f008 for ``day`` from the cache, or None if any hour is missing."""
    cyc = day.replace(hour=12)
    paths = [gfs._cache_path(cyc, fh, gfs.DEFAULT_BBOX) for fh in range(0, 9)]
    if not all(p.exists() for p in paths):
        return None
    return WindField.concat_time([WindField.load(p) for p in paths])
