import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from habpred.atmosphere import density, descent_rate
from habpred.solver import ASCENT, EARTH_RADIUS, haversine_km, predict, predict_many
from habpred.wind import (GRID_LATLON, GRID_LCC, OK, WindField, lcc_forward, lcc_params,
                          rotate_grid_to_earth, wind_at)

T0 = datetime(2025, 8, 30, 12, tzinfo=timezone.utc)


def make_field(u_fn, v_fn, hours=6, levels=40, top=45000.0):
    times = T0.timestamp() + 3600.0 * np.arange(hours)
    lats = np.arange(24.0, 33.01, 0.25)
    lons = np.arange(-88.0, -76.99, 0.25)
    lev_h = np.linspace(0.0, top, levels)
    tt, hh, yy, xx = np.meshgrid(times, lev_h, lats, lons, indexing="ij")
    hgt = hh.astype(np.float32)
    u = u_fn(tt, hh, yy, xx).astype(np.float32)
    v = v_fn(tt, hh, yy, xx).astype(np.float32)
    return WindField(GRID_LATLON, times, lats, lons, np.linspace(1000, 1, levels), hgt, u, v)


def test_density():
    assert density(0.0) == pytest.approx(1.225, rel=0.01)
    assert density(11000.0) == pytest.approx(0.364, rel=0.03)
    assert density(30000.0) == pytest.approx(0.018, rel=0.05)
    assert descent_rate(0.0, 5.0) == pytest.approx(5.0, rel=0.01)
    assert descent_rate(30000.0, 5.0) > 30.0


def test_interpolation_exact_for_linear_fields():
    def u_fn(t, h, y, x):
        return 0.001 * h + 2.0 * y - 3.0 * x + 1e-4 * (t - T0.timestamp())

    def v_fn(t, h, y, x):
        return -0.0005 * h + x

    fld = make_field(u_fn, v_fn).as_tuple()
    rng = np.random.default_rng(1)
    for _ in range(200):
        t = T0.timestamp() + rng.uniform(0, 5 * 3600)
        lat, lon, alt = rng.uniform(24, 33), rng.uniform(-88, -77), rng.uniform(0, 45000)
        u, v, st = wind_at(fld, t, lat, lon, alt)
        assert st == OK
        assert u == pytest.approx(float(u_fn(t, alt, lat, lon)), abs=2e-3)
        assert v == pytest.approx(float(v_fn(t, alt, lat, lon)), abs=2e-3)


def test_out_of_domain():
    fld = make_field(lambda t, h, y, x: 0 * h, lambda t, h, y, x: 0 * h).as_tuple()
    assert wind_at(fld, T0.timestamp(), 40.0, -80.0, 100.0)[2] != OK
    assert wind_at(fld, T0.timestamp() - 1, 28.0, -80.0, 100.0)[2] != OK


def test_vertical_profile_no_wind():
    fld = make_field(lambda t, h, y, x: 0 * h, lambda t, h, y, x: 0 * h)
    f = predict(fld, T0, 28.5, -81.2, 20.0, 5.0, 30000.0, 5.0, dt=60.0)
    assert f.status == 0
    burst_t = f.burst[0] - T0.timestamp()
    assert burst_t == pytest.approx((30000 - 20) / 5.0, abs=0.05)
    assert f.burst[3] == pytest.approx(30000.0)
    # descent time = integral of dz / v(z)
    z = np.linspace(20.0, 30000.0, 200001)
    rates = np.array([descent_rate(zz, 5.0) for zz in z[::100]])
    expected = np.trapezoid(1 / rates, z[::100])
    assert f.t[-1] - f.burst[0] == pytest.approx(expected, rel=1e-3)
    assert f.lat[-1] == pytest.approx(28.5) and f.lon[-1] == pytest.approx(-81.2)


def test_uniform_wind_drift():
    fld = make_field(lambda t, h, y, x: 10.0 + 0 * h, lambda t, h, y, x: 0 * h)
    f = predict(fld, T0, 28.0, -82.0, 0.0, 5.0, 20000.0, 5.0, dt=30.0)
    flight_s = f.t[-1] - f.t[0]
    # eastward drift at constant latitude, radius grows slightly with altitude
    dist_km = haversine_km(28.0, -82.0, f.lat[-1], f.lon[-1])
    assert f.lat[-1] == pytest.approx(28.0, abs=1e-9)
    assert dist_km == pytest.approx(10.0 * flight_s / 1000, rel=4e-3)


def test_predict_many_matches_single():
    fld = make_field(lambda t, h, y, x: 5 + 0.0004 * h, lambda t, h, y, x: -2 + 0.1 * (y - 28))
    single = predict(fld, T0, 28.5, -81.2, 20.0, 6.0, 30000.0, 5.0, dt=30.0)
    many = predict_many(fld, [T0.timestamp()] * 3, 28.5, -81.2, 20.0, 6.0,
                        [30000.0, 30000.0, 25000.0], 5.0, 20.0)
    assert many[0, 1] == pytest.approx(single.lat[-1]) and many[0, 2] == pytest.approx(single.lon[-1])
    assert many[1, 1] == many[0, 1]
    assert many[2, 4] == 0 and many[2, 0] < many[0, 0]


def test_lcc_rotation_north_wind():
    proj = lcc_params(38.5, -97.5, 6371229.0)
    lat, lon = 28.5, -81.0
    x0, y0, theta = lcc_forward(proj, lat, lon)
    x1, y1, _ = lcc_forward(proj, lat + 0.01, lon)
    # unit northward vector in grid coords
    d = np.array([x1 - x0, y1 - y0])
    d /= np.linalg.norm(d)
    ue, ve = rotate_grid_to_earth(d[0], d[1], theta)
    assert ue == pytest.approx(0.0, abs=1e-6)
    assert ve == pytest.approx(1.0, abs=1e-6)


def test_rate_profiles():
    fld = make_field(lambda t, h, y, x: 0 * h, lambda t, h, y, x: 0 * h)
    # 5 m/s to 10 km then 8 m/s (sharp step), constant 6 m/s descent table
    asc = [[0, 5.0], [10000, 5.0], [10000.001, 8.0], [40000, 8.0]]
    desc = [[0, 6.0], [40000, 6.0]]
    f = predict(fld, T0, 28.5, -81.2, 0.0, asc, 30000.0, desc, dt=10.0)
    assert f.burst[0] - T0.timestamp() == pytest.approx(10000 / 5 + 20000 / 8, abs=1.0)
    assert f.t[-1] - f.burst[0] == pytest.approx(30000 / 6, abs=0.1)
