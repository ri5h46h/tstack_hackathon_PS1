#!/usr/bin/env python3
"""
validate_trips.py  -  check our bus travel-time estimates against what buses really did.

For every real bus run on a day the model did NOT learn from, we pick pairs of stops
(A, B) on that run. The real time is arrival at B minus arrival at A. We estimate the same
time three ways and compare:

  ours        the learned table for that time of day (what the planner uses)
  all-day     the same table but one all-day number per stretch (no time-of-day effect)
  timetable   the scheduled time in GTFS stop_times (what the published schedule says)

"Did not learn from" is enforced: the table for each held-out group is rebuilt from the other
groups only (days if there are 2 or more, otherwise groups of buses), and its scale factors are
learned on days it did not see. This is the same cross-fitting used in train_eta.py.

Run from the repo root:
    uv run python eval/validate_trips.py
    uv run python eval/validate_trips.py --pairs-per-run 4 --threads 2
    uv run python eval/validate_trips.py selftest

Writes eval/validation/ : trips.csv (every case), summary.md (numbers to paste into the report),
and validation.png (charts, if matplotlib is installed).
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT))

import build_eta_bucketed as B  # noqa: E402
import train_eta as T  # noqa: E402

SPANS = (3, 5, 8, 12, 18, 25, 35)        # stops between A and B that we test
MIN_REAL_S, MAX_REAL_S = 60, 5400
SPAN_BINS = [(1, 4, "3-4 stops"), (5, 9, "5-9 stops"), (10, 19, "10-19 stops"), (20, 99, "20+ stops")]


# ---------------------------------------------------------------- timetable
FEED_TRIP = __import__("re").compile(r"^\d+_(\d{1,2})_(\d{2})_")
MATCH_WINDOW_S = 45 * 60      # a feed trip may start this far from the scheduled start
MAX_CANDIDATES = 4


def load_timetable(gtfs_dir, run_trips, log=print):
    """
    run_trips: DataFrame with columns route_id, trip_id (one row per distinct pair in the arrivals).
    Returns ({(gtfs_trip_id, stop_id): scheduled seconds}, {(route_id, feed_trip_id): [gtfs trip ids]}).

    The live feed names trips like '1000_06_13_185' (route, start hour, start minute, counter), GTFS
    names them like '1000_FULL_06_10' or '1000_SHORT_11_05'. A route has several patterns (FULL, SHORT)
    and trips at several times of day. For each feed trip we take the GTFS trips of the same route,
    nearest in start time (within 45 min), and later choose the first one that serves both stops of
    the stretch being tested. If the ids are already identical they are used as they are.
    """
    g = Path(gtfs_dir)
    if not (g / "stop_times.txt").exists():
        log(f"timetable: {(g / 'stop_times.txt').resolve()} does not exist (run from the repo root or pass --gtfs <folder>)")
        return {}, {}
    st = pd.read_csv(g / "stop_times.txt", usecols=["trip_id", "stop_id", "arrival_time"],
                     dtype={"trip_id": str, "arrival_time": str}).dropna(subset=["arrival_time"])
    hms = st["arrival_time"].str.strip().str.split(":", expand=True).astype(float)
    st["sec"] = hms[0] * 3600 + hms[1] * 60 + hms[2]
    start = st.groupby("trip_id")["sec"].min()
    cand = {}
    by_route = {}
    if (g / "trips.txt").exists():
        tr = pd.read_csv(g / "trips.txt", usecols=["route_id", "trip_id"], dtype={"trip_id": str})
        tr = tr[tr["trip_id"].isin(start.index)]
        tr["start"] = tr["trip_id"].map(start)
        by_route = {int(r): d[["start", "trip_id"]].to_numpy() for r, d in tr.groupby("route_id")}
    n_ok = 0
    pairs = run_trips.dropna().drop_duplicates()
    for r, t in zip(pairs["route_id"], pairs["trip_id"]):
        r, t = int(r), str(t)
        if t in start.index:
            cand[(r, t)] = [t]
        else:
            m = FEED_TRIP.match(t)
            arr = by_route.get(r)
            if not m or arr is None:
                continue
            want = int(m.group(1)) * 3600 + int(m.group(2)) * 60
            d = np.abs(arr[:, 0].astype(float) - want)
            order = [k for k in np.argsort(d)[:MAX_CANDIDATES] if d[k] <= MATCH_WINDOW_S]
            if order:
                cand[(r, t)] = [str(arr[k, 1]) for k in order]
        n_ok += (r, t) in cand
    if not cand:
        log(f"timetable: read {len(st):,} rows but no feed trip could be matched to a GTFS trip.\n"
            f"  feed trip_id looks like: {list(pairs['trip_id'][:3])}\n  GTFS trip_id looks like: {list(start.index[:3])}")
        return {}, {}
    used = {x for v in cand.values() for x in v}
    sub = st[st["trip_id"].isin(used)]
    tt = {}
    for t, stp, v in zip(sub["trip_id"], sub["stop_id"], sub["sec"]):
        tt.setdefault((t, int(stp)), v)
    log(f"timetable: matched {n_ok:,} of {len(pairs):,} feed trips to GTFS trips")
    return tt, cand


# ---------------------------------------------------------------- one held-out group
def group_cases(arr, lk, lk_all, tt, trip_map, pairs_per_run, rng):
    a = arr[arr["quality"] != "coarse"].sort_values(["run_key", "stop_idx"])
    rows = []
    for run, d in a.groupby("run_key", sort=False):
        idx = d["stop_idx"].values
        n = len(d)
        if n < 4 or (np.diff(idx) != 1).any():
            continue                                    # need an unbroken run of stops
        rid = int(d["route_id"].iloc[0])
        sid = d["stop_id"].values.astype("int64")
        ts = d["arrive_ts"].values.astype(float)
        prog = d["stop_prog_m"].values.astype(float)
        cands = trip_map.get((rid, str(d["trip_id"].iloc[0])), [])
        veh = d["vehicle_id"].iloc[0]
        spans = [s for s in SPANS if s < n]
        if not spans:
            continue
        for _ in range(pairs_per_run):
            span = int(rng.choice(spans))
            i = int(rng.integers(0, n - span))
            j = i + span
            real = ts[j] - ts[i]
            if not (MIN_REAL_S <= real <= MAX_REAL_S):
                continue
            hour = int(((ts[i] + B.IST) // 3600) % 24)
            bucket = lk.bucket_for(hour)
            est = est_all = 0.0
            n_pair = 0
            for k in range(i, j):
                dist = max(prog[k + 1] - prog[k], 1.0)
                sec, _, src = lk.seg_stats(rid, int(sid[k]), int(sid[k + 1]), bucket, dist, scaled=True)
                sec_a, _, _ = lk_all.seg_stats(rid, int(sid[k]), int(sid[k + 1]), "all", dist, scaled=True)
                est += sec if sec else 60.0
                est_all += sec_a if sec_a else 60.0
                n_pair += src in ("pair_blend", "pair_all")
            sched, trip = None, ""
            for c in cands:
                a0, a1 = tt.get((c, int(sid[i]))), tt.get((c, int(sid[j])))
                if a0 is not None and a1 is not None and a1 > a0:
                    sched, trip = a1 - a0, c
                    break
            rows.append({"run": int(run), "route_id": rid, "vehicle_id": veh, "gtfs_trip": trip,
                         "from_stop": int(sid[i]), "to_stop": int(sid[j]), "stops": span,
                         "dist_m": round(prog[j] - prog[i]), "hour": hour, "bucket": bucket,
                         "real_s": round(real), "ours_s": round(est), "allday_s": round(est_all),
                         "timetable_s": None if sched is None else round(sched),
                         "pair_data_share": round(n_pair / span, 2)})
    return rows


# ---------------------------------------------------------------- numbers
def metrics(df, col):
    d = df.dropna(subset=[col])
    if d.empty:
        return None
    err = d[col] - d["real_s"]
    pct = (err / d["real_s"]) * 100
    return {"n": len(d), "mae_min": err.abs().mean() / 60, "median_ae_min": err.abs().median() / 60,
            "mape": (err.abs() / d["real_s"]).mean() * 100, "median_ape": (err.abs() / d["real_s"]).median() * 100,
            "bias_pct": pct.mean(), "within_20": float(((err.abs() / d["real_s"]) <= 0.2).mean() * 100)}


def table(df, by, label, cols=("ours_s", "allday_s", "timetable_s")):
    lines = [f"| {label} | cases | " + " | ".join(f"{c.replace('_s', '')} MAPE %" for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for key, d in by:
        cells = []
        for c in cols:
            m = metrics(d, c)
            cells.append("-" if m is None else f"{m['mape']:.1f}")
        lines.append(f"| {key} | {len(d):,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(df, out, info, days, mode):
    L = ["# Validation against actual bus trips", "",
         f"Data: {len(days)} day(s) ({', '.join(days)}). Held-out by {mode}: each case is scored by a table built "
         "without that group. Real time = arrival at stop B minus arrival at stop A on a real bus run.", ""]
    L += ["## Overall", "", "| method | cases | MAE (min) | median abs error (min) | MAPE % | median APE % | bias % | within 20 % |",
          "|---|---|---|---|---|---|---|---|"]
    names = {"ours_s": "Ours (time-of-day table)", "allday_s": "All-day table (no time-of-day)",
             "timetable_s": "GTFS timetable"}
    for c, nm in names.items():
        m = metrics(df, c)
        if m:
            L.append(f"| {nm} | {m['n']:,} | {m['mae_min']:.2f} | {m['median_ae_min']:.2f} | {m['mape']:.1f} | "
                     f"{m['median_ape']:.1f} | {m['bias_pct']:+.1f} | {m['within_20']:.0f} % |")
    both = df.dropna(subset=["timetable_s"])
    if len(both) and len(both) < len(df):
        L += ["", f"Timetable is available for {len(both):,} of {len(df):,} cases. On those same cases:", "",
              "| method | MAPE % |", "|---|---|"]
        for c, nm in names.items():
            m = metrics(both, c)
            L.append(f"| {nm} | {m['mape']:.1f} |")
    df = df.copy()
    df["span_bin"] = pd.cut(df["stops"], [b[0] - 1 for b in SPAN_BINS] + [SPAN_BINS[-1][1]],
                            labels=[b[2] for b in SPAN_BINS])
    L += ["", "## By trip length", "", table(df, df.groupby("span_bin", observed=True), "trip length")]
    L += ["", "## By time of day", "", table(df, df.groupby("bucket"), "time bucket")]
    hours = df.groupby("hour")
    L += ["", "## By hour", "", table(df, hours, "hour (IST)")]
    hi = df[df["pair_data_share"] >= 0.8]
    lo = df[df["pair_data_share"] < 0.8]
    L += ["", "## Effect of data coverage", "",
          f"- cases where at least 80 % of the stretches had real stop-pair data: {len(hi):,}"
          f" (MAPE {metrics(hi, 'ours_s')['mape']:.1f} %)" if len(hi) else "- no high-coverage cases",
          f"- cases with less coverage (speed fallback used): {len(lo):,}"
          f" (MAPE {metrics(lo, 'ours_s')['mape']:.1f} %)" if len(lo) else "- no low-coverage cases"]
    L += ["", "## Notes", "",
          "- Real time includes time standing at stops and signals, because it runs from arrival at A to arrival at B.",
          "- Stop pairs are sampled at random from real runs (about equal numbers per run), not from planner output.",
          "- With one day of data, groups are sets of buses, so the same road conditions appear in learn and test data. "
          "Results on several days are the honest ones."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def plot(df, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("chart skipped: matplotlib is not installed (uv add matplotlib, then: validate_trips.py plot)")
        return False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    s = df.sample(min(len(df), 6000), random_state=1)
    lim = max(s["real_s"].max(), s["ours_s"].max()) / 60
    ax[0].scatter(s["real_s"] / 60, s["ours_s"] / 60, s=4, alpha=0.25, color="#1f7a5c")
    ax[0].plot([0, lim], [0, lim], color="#444", lw=1)
    ax[0].set(xlabel="real time (min)", ylabel="our estimate (min)", title="Estimate vs real")
    hr = df.groupby("hour")
    for col, nm, c in (("ours_s", "ours", "#1f7a5c"), ("allday_s", "all-day", "#c58a1b"), ("timetable_s", "timetable", "#b5473a")):
        y = hr.apply(lambda d: ((d[col] - d["real_s"]).abs() / d["real_s"]).mean() * 100 if d[col].notna().any() else np.nan,
                     include_groups=False)
        ax[1].plot(y.index, y.values, marker="o", ms=3, label=nm, color=c)
    ax[1].set(xlabel="hour of day (IST)", ylabel="MAPE %", title="Error by hour")
    ax[1].legend()
    err = ((df["ours_s"] - df["real_s"]) / df["real_s"] * 100).clip(-100, 150)
    ax[2].hist(err, bins=60, color="#1f7a5c")
    ax[2].axvline(0, color="#444", lw=1)
    ax[2].set(xlabel="our error (% of real time)", title="Error spread")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=130)
    return True


# ---------------------------------------------------------------- main
def run(a, log=print, bars=True):
    info_path = Path(a.data) / "eta_bucketed_info.json"
    if not info_path.exists():
        sys.exit(f"{info_path} not found: run build/build_eta_bucketed.py build first")
    import json
    final_info = json.load(open(info_path))
    buckets = [tuple(b) for b in final_info["buckets"]]
    arr_files = sorted(Path(a.data, "arrivals").glob("*_arrivals.csv.gz"))
    if not arr_files:
        sys.exit(f"No arrivals files in {a.data}/arrivals")
    day_list = [f.name[:10] for f in arr_files]
    arrs = []
    bar = T.Progress("reading arrivals", len(arr_files), bars)
    for i, f in enumerate(arr_files):
        x = pd.read_csv(f, usecols=["run_id", "route_id", "vehicle_id", "trip_id", "stop_idx", "stop_id",
                                    "arrive_ts", "quality", "stop_prog_m"], dtype={"trip_id": str})
        x["run_key"] = x["run_id"].astype("int64") + i * 10_000_000
        x["day"] = f.name[:10]
        arrs.append(x)
        bar.update()
    bar.close()
    arr = pd.concat(arrs, ignore_index=True)
    seg = B.load_segments(a.data, day_list)
    if len(day_list) >= 2:
        gmap = {d: i for i, d in enumerate(day_list)}
        seg["grp"], arr["grp"] = seg["day"].map(gmap).values, arr["day"].map(gmap).values
        n_groups, mode = len(day_list), "day"
    else:
        n_groups, mode = a.folds, "bus"
        seg["grp"] = T.vehicle_fold(seg["vehicle_id"], n_groups)
        arr["grp"] = T.vehicle_fold(arr["vehicle_id"], n_groups)
    log(f"days: {', '.join(day_list)}   groups: {n_groups} (by {mode})")
    tt, trip_map = ({}, {}) if a.no_timetable else load_timetable(a.gtfs, arr[["route_id", "trip_id"]], log)

    rows = []
    bar = T.Progress("scoring held-out trips", n_groups, bars)
    for g in range(n_groups):
        learn = seg[seg["grp"] != g]
        sb, sall, _ = B.fit_scale(learn, buckets, False) if learn["day"].nunique() >= 2 else ({}, 1.0, "")
        info = {**final_info, "scale_bucket": sb, "scale_all": sall}
        lk = T.make_lookup(learn, buckets, info)
        lk_all = lk
        rows += group_cases(arr[arr["grp"] == g], lk, lk_all, tt, trip_map, a.pairs_per_run, np.random.default_rng(7 + g))
        bar.update()
    bar.close()
    if not rows:
        sys.exit("No validation cases could be built.")
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trips.csv", index=False)
    text = write_summary(df, out, final_info, day_list, mode)
    made = plot(df, out)
    log("\n" + text)
    log(f"\nwrote {out / 'trips.csv'}, {out / 'summary.md'}" + (f", {out / 'validation.png'}" if made else ""))
    return df


def _test_matching():
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp)
        pd.DataFrame([("1_FULL_06_05", "06:05:00", 10, 1), ("1_FULL_06_05", "06:09:30", 11, 2),
                      ("1_SHORT_06_20", "06:20:00", 10, 1), ("1_SHORT_06_20", "06:23:00", 12, 2),
                      ("1_FULL_07_00", "07:00:00", 10, 1), ("1_FULL_07_00", "07:05:00", 11, 2)],
                     columns=["trip_id", "arrival_time", "stop_id", "stop_sequence"]).to_csv(g / "stop_times.txt", index=False)
        pd.DataFrame({"route_id": [1, 1, 1], "trip_id": ["1_FULL_06_05", "1_SHORT_06_20", "1_FULL_07_00"]}
                     ).to_csv(g / "trips.txt", index=False)
        rt = pd.DataFrame({"route_id": [1, 1, 1, 9], "trip_id": ["1_06_13_77", "1_09_00_1", "1_20_00_1", "9_06_05_1"]})
        tt, cand = load_timetable(g, rt, log=lambda *x: None)
        assert cand[(1, "1_06_13_77")][0] in ("1_FULL_06_05", "1_SHORT_06_20"), cand
        assert (1, "1_20_00_1") not in cand and (9, "9_06_05_1") not in cand
        assert tt[("1_FULL_06_05", 11)] - tt[("1_FULL_06_05", 10)] == 270


def _selftest():
    _test_matching()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        T._synthetic(root)
        g = Path(tmp) / "gtfs"
        g.mkdir()
        pd.DataFrame([("x", f"00:{(i * 90) // 60 + 10:02d}:{(i * 90) % 60:02d}", 100 + i, i) for i in range(40)],
                     columns=["trip_id", "arrival_time", "stop_id", "stop_sequence"]).to_csv(g / "stop_times.txt", index=False)
        seg = B.load_segments(str(root))
        bk = B.parse_buckets(B.DEFAULT_BUCKETS)
        sb, sa, _ = B.fit_scale(seg, bk, False)
        table_, info, _ = B.build_tables(seg, bk, False)
        info["scale_bucket"], info["scale_all"] = sb, sa
        table_.to_csv(root / "eta_bucketed.csv.gz", index=False)
        import json
        json.dump(info, open(root / "eta_bucketed_info.json", "w"))
        ns = argparse.Namespace(data=str(root), gtfs=str(g), out=str(Path(tmp) / "out"), pairs_per_run=4,
                                folds=3, no_timetable=False, threads=1)
        df = run(ns, log=lambda *x: None, bars=False)
        assert len(df) > 500 and (Path(tmp) / "out" / "summary.md").exists()
        mo, ma = metrics(df, "ours_s"), metrics(df, "allday_s")
        print(f"selftest: {len(df)} cases, MAPE ours {mo['mape']:.1f} %, all-day {ma['mape']:.1f} %, "
              f"timetable cases {df['timetable_s'].notna().sum()}")
        assert mo["mape"] < 40, "synthetic data should be predicted reasonably"
        print("PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "plot", "selftest"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--gtfs", default="gtfs")
    ap.add_argument("--out", default="eval/validation")
    ap.add_argument("--pairs-per-run", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3, help="bus groups when there is only one day")
    ap.add_argument("--no-timetable", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for numpy/pandas (0 = all)")
    a = ap.parse_args()
    T.limit_threads(a.threads)
    if a.cmd == "selftest":
        _selftest()
    elif a.cmd == "plot":
        out = Path(a.out)
        if not (out / "trips.csv").exists():
            sys.exit(f"{out / 'trips.csv'} not found: run  uv run eval/validate_trips.py  first")
        if plot(pd.read_csv(out / "trips.csv"), out):
            print(f"wrote {out / 'validation.png'}")
        else:
            sys.exit("matplotlib is not installed: run  uv add matplotlib  (or pip install matplotlib) and try again")
    else:
        run(a)


if __name__ == "__main__":
    main()
