#!/usr/bin/env python3
"""
build_eta_bucketed.py

Builds stop-to-stop travel times per route, split by time of day.
It reads the arrivals files made by build_arrivals.py and writes a lookup table.

How an estimate is made (see EtaLookup):
  1. bucket median and all-day median for the stop pair are blended
     (weight on the bucket = n / (n + K), K = 20 by default)
  2. if the pair has no data, distance / route speed is used
  3. the result is multiplied by a scale factor per bucket. Medians of segment
     times add up to less than real journey times (waiting at stops and signals),
     so the factor is learned on days the table did not see.

Commands
    build     make data/eta_bucketed.csv.gz and data/eta_bucketed_info.json
    check     honest test on a day not used for learning
    selftest  small synthetic test

Examples (run from the repo root)
    python build/build_eta_bucketed.py build
    python build/build_eta_bucketed.py check
    python build/build_eta_bucketed.py build --buckets "morning:6-11,midday:11-16,pm_peak:16-20,night:20-6"
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

IST = 19800
MIN_N = 3                 # fewest samples to keep a cell in the table
SEG_MIN_S, SEG_MAX_S = 3, 600
V_MIN, V_MAX = 1.0, 20.0  # m/s clip for speeds
BLEND_K = 20
SCALE_MIN_SEGS = 2000     # fewest calibration segments to trust a bucket's own scale
SCALE_CLIP = (0.8, 1.6)
DEFAULT_BUCKETS = "morning:6-11,midday:11-16,pm_peak:16-20,night:20-6"


# ---------------------------------------------------------------- buckets
def parse_buckets(spec):
    """'name:start-end,...' with hours 0-24. A bucket may wrap midnight (20-6)."""
    out = []
    for part in spec.split(","):
        name, rng = part.strip().split(":")
        a, b = rng.split("-")
        out.append((name.strip(), int(a), int(b)))
    return out


def hour_to_bucket(hours, buckets):
    """hours: numpy array of ints 0-23. Returns array of bucket names."""
    res = np.full(len(hours), "other", dtype=object)
    for name, a, b in buckets:
        if a < b:
            m = (hours >= a) & (hours < b)
        else:  # wraps midnight
            m = (hours >= a) | (hours < b)
        res[m] = name
    return res


# ---------------------------------------------------------------- loading
def load_segments(data_dir, days=None):
    """Return a DataFrame with one row per stop-to-stop segment."""
    files = sorted(Path(data_dir, "arrivals").glob("*_arrivals.csv.gz"))
    if days:
        want = set(days)
        files = [f for f in files if f.name[:10] in want]
    if not files:
        sys.exit(f"No arrivals files found in {data_dir}/arrivals")
    parts = []
    for i, f in enumerate(files):
        day = f.name[:10]
        a = pd.read_csv(
            f,
            usecols=["run_id", "route_id", "vehicle_id", "stop_idx", "stop_id",
                     "arrive_ts", "quality", "stop_prog_m"],
        )
        a = a[a["quality"] != "coarse"].copy()
        a["run_key"] = a["run_id"].astype("int64") + i * 10_000_000
        a = a.sort_values(["run_key", "stop_idx"])
        g = a.groupby("run_key", sort=False)
        prev_idx = g["stop_idx"].shift()
        ok = ((a["stop_idx"] - prev_idx) == 1).values
        prev_stop = g["stop_id"].shift()
        prev_ts = g["arrive_ts"].shift()
        prev_prog = g["stop_prog_m"].shift()
        seg = pd.DataFrame({
            "day": day,
            "run_key": a["run_key"].values[ok],
            "vehicle_id": a["vehicle_id"].values[ok],
            "route_id": a["route_id"].values[ok],
            "from_stop": prev_stop.values[ok].astype("int64"),
            "to_stop": a["stop_id"].values[ok],
            "t_from": prev_ts.values[ok],
            "t": (a["arrive_ts"] - prev_ts).values[ok],
            "dist_m": (a["stop_prog_m"] - prev_prog).values[ok],
        })
        seg = seg[(seg["t"] >= SEG_MIN_S) & (seg["t"] <= SEG_MAX_S)]
        parts.append(seg)
    seg = pd.concat(parts, ignore_index=True)
    seg["hour"] = (((seg["t_from"] + IST) // 3600) % 24).astype(int)
    dt = pd.to_datetime(seg["day"])
    seg["daytype"] = np.where(dt.dt.day_of_week >= 5, "weekend", "weekday")
    return seg


# ---------------------------------------------------------------- tables
def summarise(seg, keys):
    g = seg.groupby(keys, sort=False)
    out = pd.DataFrame({
        "n": g["t"].size(),
        "median_s": g["t"].median(),
        "p25_s": g["t"].quantile(0.25),
        "p75_s": g["t"].quantile(0.75),
        "dist_m": g["dist_m"].median(),
    }).reset_index()
    out = out[out["n"] >= MIN_N]
    for c in ("median_s", "p25_s", "p75_s", "dist_m"):
        out[c] = out[c].round(1)
    return out


def speeds(seg, keys):
    s = seg[seg["dist_m"] > 0].copy()
    s["v"] = (s["dist_m"] / s["t"]).clip(V_MIN, V_MAX)
    g = s.groupby(keys)["v"]
    r = pd.DataFrame({"n": g.size(), "v": g.median().round(2)})
    return r[r["n"] >= MIN_N]


def add_bucket(seg, buckets, split_daytype=False):
    seg = seg.copy()
    seg["bucket"] = hour_to_bucket(seg["hour"].values, buckets)
    if split_daytype:
        seg["bucket"] = seg["bucket"] + "|" + seg["daytype"]
    return seg


def build_tables(seg, buckets, split_daytype=False):
    seg = add_bucket(seg, buckets, split_daytype)
    pair = ["route_id", "from_stop", "to_stop"]
    t_bucket = summarise(seg, pair + ["bucket"])
    t_all = summarise(seg, pair)
    t_all.insert(3, "bucket", "all")
    table = pd.concat([t_bucket, t_all], ignore_index=True)

    allv = seg[seg["dist_m"] > 0]
    vg = float((allv["dist_m"] / allv["t"]).clip(V_MIN, V_MAX).median()) if len(allv) else 8.0
    v_route = speeds(seg, ["route_id"])["v"].to_dict()
    vrb = speeds(seg, ["route_id", "bucket"])["v"]
    v_route_bucket = {f"{r}|{b}": v for (r, b), v in vrb.items()}
    vgb = speeds(seg, ["bucket"])["v"].to_dict()
    info = {
        "v_global": round(vg, 2),
        "v_global_bucket": vgb,
        "v_route": {str(k): v for k, v in v_route.items()},
        "v_route_bucket": v_route_bucket,
        "buckets": [list(b) for b in buckets],
        "split_daytype": split_daytype,
        "segments_seen": int(len(seg)),
        "segment_pairs": int(len(t_all)),
        "routes": int(seg["route_id"].nunique()),
        "days": sorted(seg["day"].unique().tolist()),
        "min_n": MIN_N,
        "blend_k": BLEND_K,
        "scale_bucket": {},
        "scale_all": 1.0,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return table, info, seg


# ---------------------------------------------------------------- prediction
def predict_df(df, table, info, k=BLEND_K, scale=None):
    """
    Vectorised segment-time prediction for a DataFrame of segments that has
    route_id, from_stop, to_stop, bucket, dist_m.
    k=None: all-day median only.  k>0: blend of bucket and all-day medians.
    scale: (dict bucket->factor, default factor) or None.
    """
    key = ["route_id", "from_stop", "to_stop"]
    tb = table[table["bucket"] != "all"][key + ["bucket", "median_s", "n"]]
    tb = tb.rename(columns={"median_s": "mb", "n": "nb"})
    ta = table[table["bucket"] == "all"][key + ["median_s"]].rename(columns={"median_s": "ma"})
    d = df[key + ["bucket", "dist_m"]].reset_index(drop=True)
    d = d.merge(tb, on=key + ["bucket"], how="left").merge(ta, on=key, how="left")

    rid = d["route_id"].astype(str)
    v = (rid + "|" + d["bucket"]).map(info["v_route_bucket"])
    v = v.fillna(rid.map(info["v_route"]))
    v = v.fillna(d["bucket"].map(info["v_global_bucket"]))
    v = v.fillna(info["v_global"]).astype(float)
    dist = d["dist_m"].astype(float).values
    fb = np.where(dist > 0, dist / v.values, 60.0)

    ma = d["ma"].values.astype(float)
    mb = d["mb"].values.astype(float)
    nb = d["nb"].fillna(0).values.astype(float)
    base = np.where(np.isnan(ma), fb, ma)
    if k:
        w = nb / (nb + k)
        p = np.where(np.isnan(mb), base, w * mb + (1 - w) * base)
    else:
        p = base
    if scale is not None:
        sb, sall = scale
        f = d["bucket"].map(sb).fillna(sall).values.astype(float)
        p = p * f
    return p


def fit_scale(seg, buckets, split_daytype, k=BLEND_K, n_folds=3):
    """
    Learn the scale factor on days the table did not see: for each of the last
    n_folds days, build the table from the other days and compare with that day.
    Returns (dict bucket->factor, overall factor, text report).
    """
    days = sorted(seg["day"].unique())
    if len(days) < 2:
        return {}, 1.0, "only one day: scale not learned (factor 1.0)"
    folds = days[-min(n_folds, len(days) - 1):] if len(days) > n_folds else days[1:]
    act, pred, bks = [], [], []
    for d in folds:
        learn, cal = seg[seg["day"] != d], seg[seg["day"] == d]
        table, info, _ = build_tables(learn, buckets, split_daytype)
        cal = add_bucket(cal, buckets, split_daytype)
        pred.append(predict_df(cal, table, info, k))
        act.append(cal["t"].values.astype(float))
        bks.append(cal["bucket"].values)
    act, pred, bks = np.concatenate(act), np.concatenate(pred), np.concatenate(bks)
    lo, hi = SCALE_CLIP
    sall = float(np.clip(act.sum() / pred.sum(), lo, hi))
    sb = {}
    lines = [f"scale learned on days {', '.join(folds)}: overall {sall:.3f}"]
    for b in sorted(set(bks)):
        m = bks == b
        if m.sum() >= SCALE_MIN_SEGS:
            sb[b] = float(np.clip(act[m].sum() / pred[m].sum(), lo, hi))
            lines.append(f"  {b:<12} {sb[b]:.3f}  ({int(m.sum()):,} segments)")
        else:
            lines.append(f"  {b:<12} too few segments ({int(m.sum())}), uses overall")
    return sb, sall, "\n".join(lines)


# ---------------------------------------------------------------- lookup
class EtaLookup:
    """
    Segment time lookup with a fallback chain. Every answer says where it came from.

    source          meaning                                    confidence
    pair_blend      stop pair: bucket and all-day mixed        high
    pair_all        stop pair: all-day median                  medium
    route_bucket    distance / route speed in this bucket      low
    route_all       distance / route speed, all day            low
    global_bucket   distance / network speed in this bucket    very low
    global          distance / network speed                   very low

    seg_time() multiplies by the scale factor of the bucket (the planner's numbers).
    seg_stats(scaled=False) gives the raw blend plus its spread (the ETA model's input).
    """

    def __init__(self, table_path=None, info_path=None, blend_k=None, table=None, info=None):
        if table is None:
            table = pd.read_csv(table_path)
            info = json.load(open(info_path))
        self.info = info
        self.buckets = [tuple(b) for b in self.info["buckets"]]
        self.split = self.info.get("split_daytype", False)
        self.k = self.info.get("blend_k", BLEND_K) if blend_k is None else blend_k
        self.scale_b = self.info.get("scale_bucket", {})
        self.scale_all = self.info.get("scale_all", 1.0)
        self.t = {(r, f, to, b): (m, d, n, lo, hi) for r, f, to, b, m, d, n, lo, hi in zip(
            table["route_id"], table["from_stop"], table["to_stop"], table["bucket"],
            table["median_s"], table["dist_m"], table["n"], table["p25_s"], table["p75_s"])}

    @classmethod
    def from_frames(cls, table, info, blend_k=None):
        return cls(table=table, info=info, blend_k=blend_k)

    def bucket_for(self, hour, daytype=None):
        b = hour_to_bucket(np.array([hour]), self.buckets)[0]
        return f"{b}|{daytype}" if self.split and daytype else b

    def scale(self, bucket):
        return self.scale_b.get(bucket, self.scale_all)

    def speed_fallback(self, route, bucket, dist_m):
        if dist_m is None or not dist_m > 0:
            return None, "no_data"
        i = self.info
        v = i["v_route_bucket"].get(f"{route}|{bucket}")
        if v:
            return dist_m / v, "route_bucket"
        v = i["v_route"].get(str(route))
        if v:
            return dist_m / v, "route_all"
        v = i["v_global_bucket"].get(bucket)
        if v:
            return dist_m / v, "global_bucket"
        return dist_m / i["v_global"], "global"

    def seg_stats(self, route, a, b, bucket, dist_m=None, scaled=False):
        """Return (seconds, spread_seconds, source). Spread is the p75 - p25 of segment times."""
        hb = self.t.get((route, a, b, bucket))
        ha = self.t.get((route, a, b, "all"))
        if ha:
            if hb and self.k:
                w = hb[2] / (hb[2] + self.k)
                sec = w * hb[0] + (1 - w) * ha[0]
                spr = w * (hb[4] - hb[3]) + (1 - w) * (ha[4] - ha[3])
                src = "pair_blend"
            else:
                sec, spr, src = ha[0], ha[4] - ha[3], "pair_all"
        else:
            sec, src = self.speed_fallback(route, bucket, dist_m)
            if sec is None:
                return None, None, src
            spr = 0.5 * sec
        if scaled:
            f = self.scale(bucket)
            sec, spr = sec * f, spr * f
        return sec, spr, src

    def seg_time(self, route, a, b, bucket, dist_m=None):
        """Return (seconds, source) with the bucket's scale factor applied."""
        sec, _, src = self.seg_stats(route, a, b, bucket, dist_m, scaled=True)
        return sec, src


# ---------------------------------------------------------------- commands
def cmd_build(a):
    buckets = parse_buckets(a.buckets)
    days = a.days.split(",") if a.days else None
    seg = load_segments(a.data, days)
    sb, sall, rep = fit_scale(seg, buckets, a.split_daytype)
    table, info, seg = build_tables(seg, buckets, a.split_daytype)
    info["scale_bucket"], info["scale_all"] = sb, sall
    out = Path(a.data)
    table.to_csv(out / "eta_bucketed.csv.gz", index=False)
    json.dump(info, open(out / "eta_bucketed_info.json", "w"), indent=1)
    print(f"days: {', '.join(info['days'])}")
    print(f"segments: {info['segments_seen']:,}   routes: {info['routes']:,}")
    print(f"table rows: {len(table):,}   (stop pairs with all-day median: {info['segment_pairs']:,})")
    print("\nsegments and table cells per bucket")
    cnt = seg["bucket"].value_counts()
    cells = table[table["bucket"] != "all"]["bucket"].value_counts()
    for name in sorted(cnt.index):
        print(f"  {name:<22} segments {cnt[name]:>9,}   cells(n>={MIN_N}) {cells.get(name, 0):>8,}")
    print(f"\nnetwork speed {info['v_global']} m/s")
    print(rep)
    print(f"\nwrote {out/'eta_bucketed.csv.gz'} and {out/'eta_bucketed_info.json'}")


def cmd_check(a):
    """Learn on all days but the last, test on the last day.
    The scale factor is learned inside the learning days only, so the test day stays unseen.
    With one day only: hold out 20% of vehicles (by hash) and say so."""
    buckets = parse_buckets(a.buckets)
    days = a.days.split(",") if a.days else None
    seg = load_segments(a.data, days)
    all_days = sorted(seg["day"].unique())
    if len(all_days) >= 2:
        test_mask = seg["day"] == all_days[-1]
        print(f"learn on {len(all_days)-1} day(s), test on {all_days[-1]}")
    else:
        h = pd.util.hash_array(seg["vehicle_id"].astype(str).values) % 5
        test_mask = pd.Series(h == 0, index=seg.index)
        print("ONLY ONE DAY: testing on 20% of vehicles held out. This is easier than a new day.")
    train, test = seg[~test_mask], seg[test_mask]

    sb, sall, rep = fit_scale(train, buckets, False)
    print(rep)
    table, info, _ = build_tables(train, buckets, False)
    test = add_bucket(test, buckets).sort_values(["run_key", "t_from"]).reset_index(drop=True)
    n = len(test)
    actual = test["t"].values.astype(float)

    configs = [
        ("route speed only", None, "speed"),
        ("all-day median", None, 0),
        ("all-day + scale", (sb, sall), 0),
        ("blend k=10", None, 10),
        ("blend k=20", None, 20),
        ("blend k=40", None, 40),
        ("blend k=10 + scale", (sb, sall), 10),
        ("blend k=20 + scale", (sb, sall), 20),
        ("blend k=40 + scale", (sb, sall), 40),
    ]

    rk, fs, ts_ = test["run_key"].values, test["from_stop"].values, test["to_stop"].values
    L = 5
    ok = np.ones(n - L + 1, dtype=bool)
    for j in range(1, L):
        ok &= (rk[j:n - L + 1 + j] == rk[:n - L + 1]) & (fs[j:n - L + 1 + j] == ts_[j - 1:n - L + j])

    def chain(p):
        return sum(p[j:n - L + 1 + j] for j in range(L))

    act5 = chain(actual)[ok]
    print(f"\ntest segments: {n:,}   5-stop chains: {int(ok.sum()):,}   mean 5-stop time: {act5.mean():.0f} s")
    print("\nMean absolute error in seconds (same test rows for every method)")
    print(f"  {'method':<22}{'1 stop':>9}{'5 stops':>10}{'5-stop bias':>13}")
    best = None
    for name, sc, k in configs:
        if k == "speed":
            d = test.copy()
            v = (d["route_id"].astype(str)).map(info["v_route"]).fillna(info["v_global"]).astype(float)
            p = np.where(d["dist_m"] > 0, d["dist_m"] / v.values, 60.0)
        else:
            p = predict_df(test, table, info, k, sc)
        e1 = np.abs(p - actual).mean()
        p5 = chain(p)[ok]
        e5 = np.abs(p5 - act5).mean()
        bias = (p5 - act5).mean()
        print(f"  {name:<22}{e1:>9.1f}{e5:>10.1f}{bias:>13.1f}")
        if k != "speed" and (best is None or e5 < best[0]):
            best = (e5, name)
    print(f"\nBest on 5-stop error: {best[1]}")
    print("Bias < 0 means predictions are too fast. The build command uses 'blend k=20 + scale'.")


def cmd_selftest(a):
    rng = np.random.default_rng(1)
    rows = []
    for h, base in ((9, 60.0), (19, 90.0)):
        for _ in range(60):
            rows.append((1, 10, 11, h, base + rng.normal(0, 3), 500.0))
    seg = pd.DataFrame(rows, columns=["route_id", "from_stop", "to_stop", "hour", "t", "dist_m"])
    seg["day"] = "2026-10-26"
    seg["daytype"] = "weekday"
    seg["run_key"] = 0
    seg["vehicle_id"] = "x"
    seg["t_from"] = 0.0
    table, info, s2 = build_tables(seg, parse_buckets(DEFAULT_BUCKETS))
    got = table[table["bucket"] != "all"].set_index("bucket")["median_s"].to_dict()
    p = predict_df(s2, table, info, 20)
    ok = abs(got.get("morning", 0) - 60) < 3 and abs(got.get("pm_peak", 0) - 90) < 3
    ok = ok and abs(p[0] - 60) < 15 and abs(p[-1] - 90) < 15
    print("buckets:", got, " blended examples:", round(float(p[0]), 1), round(float(p[-1]), 1))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "check", "selftest"])
    p.add_argument("--data", default="data")
    p.add_argument("--days", help="comma list like 2026-11-06,2026-11-07 (default: all)")
    p.add_argument("--buckets", default=DEFAULT_BUCKETS)
    p.add_argument("--split-daytype", action="store_true",
                   help="separate weekday and weekend cells (needs enough days of each)")
    a = p.parse_args()
    {"build": cmd_build, "check": cmd_check, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
