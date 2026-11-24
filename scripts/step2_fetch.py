"""Step 2 data fetch: radiosonde flights and the GFS forecasts needed to re-run them.

Stages (each one resumable; finished downloads are skipped on restart):

1. ``sondes``: summaries and full tracks of Florida radiosonde flights for the
   chosen months (SondeHub archive).
2. ``feb``: GFS 12Z cycle, f000-f008, for every February day of the chosen years.
   Near-analysis winds for a quick February drift climatology.
3. ``sonde-gfs``: for every sonde flight, the GFS forecast hours covering the
   flight from cycles issued about 3-9 h, 27-33 h and 51-57 h before launch.

Usage (defaults are the plan's first batch):
    uv run python scripts/step2_fetch.py [--stages sondes feb sonde-gfs]
        [--months 2026-02 2026-01 2026-03 2025-12] [--feb-years 2026 2025 2024]
        [--min-free-gb 8] [--parallel 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from habpred.sources import gfs, sondes
from habpred.sources.common import data_dir

log = logging.getLogger("step2")
LEAD_CLASSES_H = (3, 27, 51)  # cycle is the latest 6-hourly run at least this long before launch
FLIGHT_HOURS = 3.5  # sonde flights last ~2-2.5 h; cover launch-1h .. launch+3.5h


class DiskFull(RuntimeError):
    pass


def check_disk(min_free_gb: float):
    free = shutil.disk_usage(data_dir()).free / 1e9
    if free < min_free_gb:
        raise DiskFull(f"only {free:.1f} GB free (minimum {min_free_gb} GB); stopping")


def stage_sondes(months):
    out = data_dir() / "sondes" / "summaries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            s = json.loads(line)
            done[s["serial"]] = s
    for y, m in months:
        log.info("sondes: listing %04d-%02d", y, m)
        new = sondes.summaries(sondes.FLORIDA_SITES, [(y, m)])
        for s in new:
            done[s["serial"]] = s
        out.write_text("".join(json.dumps(s) + "\n" for s in done.values()))
        log.info("sondes: %d summaries for %04d-%02d (%d total)", len(new), y, m, len(done))
        serials = [s["serial"] for s in new]
        ok = 0
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(sondes.fetch_track, ser): ser for ser in serials}
            for f in as_completed(futs):
                try:
                    f.result()
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("sondes: track %s failed: %s", futs[f], e)
        log.info("sondes: %d/%d tracks for %04d-%02d", ok, len(serials), y, m)
    return list(done.values())


def launch_estimate(s) -> datetime:
    """Approximate launch time: first frame minus its altitude at ~5 m/s."""
    t = datetime.fromisoformat(s["first_time"].replace("Z", "+00:00"))
    return t - timedelta(seconds=max(0.0, s["first_alt"]) / 5.0)


def plan_sonde_gfs(summaries) -> list[tuple[datetime, int]]:
    jobs = set()
    for s in summaries:
        launch = launch_estimate(s)
        v0 = launch - timedelta(hours=1)
        v1 = launch + timedelta(hours=FLIGHT_HOURS)
        for lead in LEAD_CLASSES_H:
            c = launch - timedelta(hours=lead)
            cycle = c.replace(minute=0, second=0, microsecond=0) - timedelta(hours=c.hour % 6)
            f0 = int((v0 - cycle).total_seconds() // 3600)
            f1 = -int(-(v1 - cycle).total_seconds() // 3600)
            for fh in range(max(0, f0), min(120, f1) + 1):
                jobs.add((cycle, fh))
    return sorted(jobs, key=lambda j: (j[0] + timedelta(hours=j[1]), j[0]))


def plan_feb(years) -> list[tuple[datetime, int]]:
    jobs = []
    for y in years:
        d = datetime(y, 2, 1, 12, tzinfo=timezone.utc)
        while d.month == 2:
            jobs += [(d, fh) for fh in range(0, 9)]
            d += timedelta(days=1)
    return jobs


def run_gfs(name, jobs, parallel, min_free_gb):
    total = len(jobs)
    todo = [j for j in jobs if not gfs._cache_path(j[0], j[1], gfs.DEFAULT_BBOX).exists()]
    log.info("%s: %d forecast hours planned, %d already done, %d to fetch (~%.0f GB download)",
             name, total, total - len(todo), len(todo), len(todo) * 0.14)
    t0 = time.time()
    done = failed = 0

    def one(job):
        check_disk(min_free_gb)
        gfs.download_hour(*job)

    with ThreadPoolExecutor(parallel) as ex:
        futs = {ex.submit(one, j): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                f.result()
                done += 1
            except DiskFull:
                raise
            except FileNotFoundError as e:
                failed += 1
                log.warning("%s: missing on AWS: %s", name, e)
            except Exception as e:  # noqa: BLE001
                failed += 1
                log.warning("%s: %s f%03d failed: %r", name, j[0].strftime("%Y%m%d%H"), j[1], e)
            n = done + failed
            if n % 25 == 0 or n == len(todo):
                rate = done / max(1e-9, time.time() - t0)
                eta = (len(todo) - n) / max(rate, 1e-9) / 3600
                log.info("%s: %d/%d done, %d failed, %.2f h/s, ETA %.1f h",
                         name, n, len(todo), failed, rate, eta)
    log.info("%s: finished: %d fetched, %d failed", name, done, failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["sondes", "feb", "sonde-gfs"])
    ap.add_argument("--months", nargs="+", default=["2026-02", "2026-01", "2026-03", "2025-12"])
    ap.add_argument("--feb-years", nargs="+", type=int, default=[2026, 2025, 2024])
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--log-file", help="also append the log to this file")
    a = ap.parse_args()
    handlers = [logging.StreamHandler(sys.stdout)]
    if a.log_file:
        handlers.append(logging.FileHandler(a.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s")
    months = [tuple(int(x) for x in m.split("-")) for m in a.months]
    log.info("data dir %s", data_dir())

    try:
        summ = None
        if "sondes" in a.stages:
            summ = stage_sondes(months)
        if "feb" in a.stages:
            run_gfs("feb", plan_feb(a.feb_years), a.parallel, a.min_free_gb)
        if "sonde-gfs" in a.stages:
            if summ is None:
                p = data_dir() / "sondes" / "summaries.jsonl"
                summ = [json.loads(l) for l in p.read_text().splitlines()]
            want = {(y, m) for y, m in months}
            summ = [s for s in summ if (launch_estimate(s).year, launch_estimate(s).month) in want]
            # February flights first so the most useful results arrive soonest
            summ.sort(key=lambda s: (launch_estimate(s).month != 2, s["first_time"]))
            run_gfs("sonde-gfs", plan_sonde_gfs(summ), a.parallel, a.min_free_gb)
    except DiskFull as e:
        log.error("%s", e)
        sys.exit(3)
    log.info("all stages complete")


if __name__ == "__main__":
    main()
