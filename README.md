# hab-predictor

High altitude balloon flight prediction tools from the Knights Satellite Club.

The flight model matches [Tawhiri](https://github.com/cuspaceflight/tawhiri), the
predictor behind SondeHub: horizontal motion follows the forecast wind, ascent is at
a set rate to a burst altitude, and descent speed scales with air density. It is
written from the published equations (not Tawhiri's code) in Python and Numba. It adds:

- GFS 0.25 deg hourly winds (49 levels up to 1 hPa), and HRRR 3 km below 20 km
  blended into GFS above;
- ascent and descent rates that can vary with altitude (tables), for hindcasts from
  logged flights and for balloon physics later;
- a parallel batch solver for Monte Carlo runs and launch-site grids.

Plan and research notes: KAOS-2 Notion, "HAB landing prediction: research and plan".

## Setup

```sh
uv sync --all-extras
uv run pytest
```

Downloaded weather data is cached under `data/` (override with `HABPRED_DATA`).
A GFS forecast hour cropped to Florida is ~1 MB on disk (~140 MB downloaded); an
HRRR hour is ~75 MB on disk (~76 MB downloaded). Data comes from NOAA's open data
buckets on AWS (`noaa-gfs-bdp-pds`, `noaa-hrrr-bdp-pds`), which hold years of archive.

## Predict a flight

```sh
uv run habpred predict --lat 28.6134 --lon -81.1742 --alt 20 \
    --time 2026-11-26T15:00Z --ascent 5 --burst 30000 --descent 5 [--hrrr] [--csv out.csv]
```

`--descent` is the sea-level descent rate. `--cycle` picks a GFS run (default: latest).

From Python:

```python
from habpred.sources import gfs
from habpred.solver import predict, predict_many

wind = gfs.load(cycle, launch_time, launch_time + timedelta(hours=5))
flight = predict(wind, launch_time, lat, lon, launch_alt, 5.0, 30000.0, 5.0)
landings = predict_many(wind, t0s, lats, lons, alt, ascent, burst, descent, ground)
```

## Validation so far

`scripts/kaos1_hindcast.py`: KAOS-1 (2025-08-30) re-run with the forecasts that
were available 2 to 80 hours before launch, using the measured ascent and descent
profile (SPOT track and onboard pressure log). Landing errors: GFS 1.7 to 7.3 km,
HRRR+GFS 1.5 to 4.0 km (2 to 20 h lead). KAOS-1 drifted only 16 km, so this is one
calm summer case, not a statistic.

`scripts/compare_sondehub.py`: same inputs as the SondeHub predictor on the same GFS
cycle. With Tawhiri-like settings (0.5 deg, 3-hourly, 60 s steps) landings agree to
1.1 to 1.6 km over 135 to 189 km of drift (about 1%), with identical flight times.

## Layout

| Path | What |
|---|---|
| `src/habpred/wind.py` | Wind field container and 4-D interpolation (lat/lon and Lambert grids) |
| `src/habpred/solver.py` | RK4 flight integration, single and parallel batch |
| `src/habpred/atmosphere.py` | Density model for descent-rate scaling |
| `src/habpred/sources/` | GFS and HRRR downloaders (byte-range GRIB from AWS) |
| `src/habpred/sondehub.py` | SondeHub API client, for comparisons only |
| `scripts/` | Hindcasts and comparisons |

## Known limits

- Ground is a constant altitude (no elevation model yet). Florida is flat, so this
  costs tens of metres.
- No uncertainty estimate yet: one deterministic path per forecast.
