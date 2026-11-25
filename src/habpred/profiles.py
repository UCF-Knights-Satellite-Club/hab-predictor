"""Flight profiles from radiosonde (or tracker) tracks: cleaning, rate tables by
altitude, and the burst/last-fix points used for scoring predictions."""

from __future__ import annotations

import numpy as np

from .atmosphere import descent_rate


def clean_track(t, lat, lon, alt):
    o = np.argsort(t)
    t, lat, lon, alt = t[o], lat[o], lon[o], alt[o]
    # drop altitude spikes: points far from the running median of their neighbours
    k = 7
    pad = np.pad(alt, k // 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    keep = np.abs(alt - med) < 300.0
    return t[keep], lat[keep], lon[keep], alt[keep]


def rate_table(t, alt, sign, bin_m=1000.0):
    """[(altitude, rate)] from a monotonic-ish segment, using a fit per altitude bin."""
    rows = []
    edges = np.arange(np.floor(alt.min() / bin_m) * bin_m, alt.max() + bin_m, bin_m)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (alt >= lo) & (alt < hi)
        if m.sum() < 5 or np.ptp(t[m]) < 10:
            continue
        rate = sign * np.polyfit(t[m], alt[m], 1)[0]
        if 0.5 < rate < 150:
            rows.append([(lo + hi) / 2, rate])
    return rows


def flight_profile(t, lat, lon, alt):
    """Split a cleaned track into ascent/descent and derive rate tables. None if unusable."""
    ib = int(np.argmax(alt))
    if alt[ib] < 15000 or ib < 20:
        return None
    asc_t, asc_z = t[: ib + 1], alt[: ib + 1]
    des_t, des_z = t[ib:], alt[ib:]
    prof = {
        "t0": t[0], "lat0": lat[0], "lon0": lon[0], "alt0": alt[0],
        "burst_t": t[ib], "burst_lat": lat[ib], "burst_lon": lon[ib], "burst_alt": alt[ib],
        "asc_table": rate_table(asc_t, asc_z, +1),
        "asc_mean": (alt[ib] - alt[0]) / max(1.0, t[ib] - t[0]),
    }
    has_descent = len(des_t) > 20 and alt[ib] - des_z.min() > 3000
    if has_descent:
        j = int(np.argmin(des_z))
        prof.update(end_t=des_t[j], end_lat=lat[ib + j], end_lon=lon[ib + j], end_alt=des_z[j],
                    des_table=rate_table(des_t[: j + 1], des_z[: j + 1], -1))
        # sea-level rate for the density model matching the heard part of the descent
        z = np.linspace(des_z[j], alt[ib], 400)
        target = des_t[j] - t[ib]
        lo, hi = 1.0, 30.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            dur = np.trapezoid([1 / descent_rate(zz, mid) for zz in z], z)
            lo, hi = (mid, hi) if dur > target else (lo, mid)
        prof["des_v0"] = 0.5 * (lo + hi)
    if len(prof["asc_table"]) < 5:
        return None
    return prof
