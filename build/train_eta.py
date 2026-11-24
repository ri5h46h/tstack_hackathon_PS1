#!/usr/bin/env python3
"""
train_eta.py  -  train the live ETA correction model from your own collected days.

What it learns
    For a bus that is `h` stops before a stop, the history ETA (table time x recent live ratio)
    is wrong by some amount. The model predicts that amount (the residual)
    from: the history ETA, how far away, how spread-out the table times are, the time of day,
    how slow/fast the network, the route and the bus itself have been recently, and whether the
    bus is standing still. The features are defined in planner/features.py, which the live
    code uses too, so training and serving see the same numbers.

Why the tables are "cross-fitted"
    The ETA table was built from the same days the model learns from. If the model saw
    table values that already contain its own trips, the history ETA would look better than it
    really is, and the model would learn to trust it too much. So the data is split into groups
    (days if there are 2 or more days, otherwise buses) and the table for each group is built
    from all the OTHER groups.

Inputs (in data/):
    arrivals/{day}_arrivals.csv.gz     from build/build_arrivals.py
    pings/{day}_pings.csv.gz           from build/build_arrivals.py (any sub folder is searched)
    eta_bucketed_info.json             from build/build_eta_bucketed.py build (for the scale factors)

Output (in models/):
    eta_model.joblib, eta_model_info.json

Commands (run from the repo root)
    python build/train_eta.py train
    python build/train_eta.py train --max-cases 1000000 --step 4
    python build/train_eta.py check
    python build/train_eta.py selftest
"""
import argparse
import json
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
from planner import features as F  # noqa: E402

MAX_ACTUAL_S = 5400
SEED = 7


class Progress:
    """Small live progress bar (no extra package). Safe to call from several threads."""

    def __init__(self, label, total, enabled=True):
        import threading
        self.label, self.total, self.done = label, max(float(total), 1e-9), 0.0
        self.t0, self.last = time.time(), 0.0
        self.lock = threading.Lock()
        self.tty = sys.stderr.isatty()
        self.enabled = enabled
        self.next_pct = 10
        self._draw(force=True)

    def update(self, n=1.0):
        with self.lock:
            self.done = min(self.total, self.done + n)
            self._draw()

    def _draw(self, force=False):
        if not self.enabled:
            return
        now = time.time()
        frac = self.done / self.total
        if self.tty:
            if not force and now - self.last < 0.1 and frac < 1:
                return
            self.last = now
            el = now - self.t0
            eta = el * (1 - frac) / frac if frac > 0.01 else float("nan")
            w = 28
            bar = "#" * int(w * frac) + "-" * (w - int(w * frac))
            sys.stderr.write(f"\r{self.label:<26} [{bar}] {frac * 100:5.1f}%  {el:5.0f}s elapsed"
                             f"{'' if eta != eta else f', ~{eta:.0f}s left'}   ")
            sys.stderr.flush()
        elif frac * 100 >= self.next_pct:          # not a terminal (log file): a line every 10 %
            self.next_pct = int(frac * 10) * 10 + 10
            print(f"{self.label}: {frac * 100:.0f}%  ({now - self.t0:.0f}s)", flush=True)

    def close(self):
        with self.lock:
            self.done = self.total
            self._draw(force=True)
            if self.enabled and self.tty:
                sys.stderr.write("\n")
                sys.stderr.flush()


def limit_threads(n):
    """Cap the CPU threads used by the model code (OpenMP / BLAS). 0 = leave as is."""
    if n and n > 0:
        try:
            from threadpoolctl import threadpool_limits
            return threadpool_limits(limits=n)
        except Exception:  # noqa: BLE001  (not installed: fall back to env vars, best effort)
            import os
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[k] = str(n)
    return None


# ---------------------------------------------------------------- loading
def find_days(data_dir):
    d = Path(data_dir)
    days = sorted(f.name[:10] for f in (d / "arrivals").glob("*_arrivals.csv.gz"))
    out = []
    for day in days:
        pf = sorted((d / "arrivals").glob(f"*{day}_pings.csv.gz")) or sorted(d.rglob(f"*{day}_pings.csv.gz"))
        if pf:
            out.append((day, d / "arrivals" / f"{day}_arrivals.csv.gz", pf[0]))
    return out


def vehicle_fold(vehicle_ids, n):
    h = pd.util.hash_pandas_object(pd.Series(np.asarray(vehicle_ids).astype(str)), index=False).values
    return (h % n).astype(int)


def load_day(day, arr_path, ping_path, offset):
    arr = pd.read_csv(arr_path, usecols=["run_id", "route_id", "vehicle_id", "stop_idx", "stop_id",
                                         "arrive_ts", "quality", "stop_prog_m"])
    arr["run_key"] = arr["run_id"].astype("int64") + offset
    arr["day"] = day
    pg = pd.read_csv(ping_path, usecols=["run_id", "route_id", "vehicle_id", "ts", "prog"])
    pg["run_key"] = pg["run_id"].astype("int64") + offset
    pg["day"] = day
    return arr, pg


def route_geometry(arr):
    """{route_id: (stop_idx array, stop_id array, stop_prog array)} for routes with an unbroken stop list."""
    g = arr.groupby(["route_id", "stop_idx"]).agg(stop_id=("stop_id", "first"),
                                                  prog=("stop_prog_m", "median")).reset_index()
    out = {}
    for rid, d in g.groupby("route_id"):
        d = d.sort_values("stop_idx")
        idx = d["stop_idx"].values
        if len(idx) < 4 or (np.diff(idx) != 1).any():
            continue
        prog = np.maximum.accumulate(d["prog"].values.astype(float))
        if prog[-1] - prog[0] < 500:
            continue
        out[int(rid)] = (idx, d["stop_id"].values.astype("int64"), prog)
    return out


# ---------------------------------------------------------------- lookup for a group
def make_lookup(seg_learn, buckets, final_info):
    table, info, _ = B.build_tables(seg_learn, buckets, False)
    info["scale_bucket"] = final_info.get("scale_bucket", {})
    info["scale_all"] = final_info.get("scale_all", 1.0)
    return B.EtaLookup.from_frames(table, info)


def route_cums(lk, rid, stop_ids, prog, bucket):
    """Cumulative table time at each stop of a route (len n): scaled, raw, spread."""
    sc = lk.scale(bucket)
    n = len(stop_ids)
    w, t, sp = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)
    for i in range(n - 1):
        sec, spr, _ = lk.seg_stats(rid, int(stop_ids[i]), int(stop_ids[i + 1]), bucket,
                                   float(max(prog[i + 1] - prog[i], 1.0)), scaled=False)
        if not sec:
            sec, spr = 60.0 / sc if sc else 60.0, 30.0
        w[i], t[i], sp[i] = sec, sec * sc, spr * sc
    z = np.zeros(1)
    return np.concatenate([z, np.cumsum(t)]), np.concatenate([z, np.cumsum(w)]), np.concatenate([z, np.cumsum(sp)])


# ---------------------------------------------------------------- cases for one group
def window_sums(minutes, obs, exp, m_case, win):
    """Sums of obs/exp over events whose minute is in [m-win, m-1]. `minutes` must be sorted."""
    co = np.concatenate([[0.0], np.cumsum(obs)])
    ce = np.concatenate([[0.0], np.cumsum(exp)])
    hi = np.searchsorted(minutes, m_case, "left")
    lo = np.searchsorted(minutes, m_case - win, "left")
    return co[hi] - co[lo], ce[hi] - ce[lo]


def group_cases(pg, arr, geom, lk, buckets, step, rng, max_cases, tick=None):
    """Training rows for the pings of one group. Returns dict of arrays or None."""
    pg = pg[pg["route_id"].isin(geom)].sort_values(["run_key", "ts"]).reset_index(drop=True)
    if pg.empty:
        return None
    n = len(pg)
    run = pg["run_key"].values
    ts = pg["ts"].values.astype(float)
    prog = pg["prog"].values.astype(float)
    rid = pg["route_id"].values.astype(int)
    first = np.r_[True, run[1:] != run[:-1]]
    p_prev = np.r_[0.0, prog[:-1]]
    t_prev = np.r_[0.0, ts[:-1]]
    dt = np.where(first, 0.0, ts - t_prev)
    total = np.array([geom[r][2][-1] for r in rid])
    ok = (~first) & F.interval_ok(dt, p_prev, prog, total)
    hour = ((ts + 19800) // 3600 % 24).astype(int)
    bucket = B.hour_to_bucket(hour, buckets)

    # expected time of each ping interval, and cumulative tables per (route, bucket)
    cums = {}
    exp = np.zeros(n)
    for (r, b), idx in pd.Series(np.arange(n)).groupby([rid, bucket]).groups.items():
        idx = np.asarray(idx)
        if (r, b) not in cums:
            _, ids, sp = geom[r]
            cums[(r, b)] = route_cums(lk, r, ids, sp, b)
        cs = cums[(r, b)][0]
        sp = geom[r][2]
        e = np.interp(prog[idx], sp, cs) - np.interp(p_prev[idx], sp, cs)
        exp[idx] = np.maximum(e, 0.0)
    obs = np.where(ok, dt, 0.0)
    exp = np.where(ok, exp, 0.0)

    # network / route / own ratios
    minute = (ts // 60).astype(np.int64)
    ev = np.flatnonzero(ok)
    order = ev[np.argsort(minute[ev], kind="stable")]
    net_m, net_o, net_e = minute[order], obs[order], exp[order]
    route_ev = {}
    ro_ids = rid[order]
    for r in np.unique(ro_ids):
        route_ev[int(r)] = order[ro_ids == r]
    co, ce = np.cumsum(obs), np.cumsum(exp)
    run_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    lo = np.maximum(np.arange(n) - F.OWN_K + 1, run_start)
    own_o = co - np.where(lo > 0, co[lo - 1], 0.0)
    own_e = ce - np.where(lo > 0, ce[lo - 1], 0.0)
    standing = F.standing_seconds(run, ts, prog)

    # which pings become cases
    keep = np.zeros(n, dtype=bool)
    keep[::max(1, step)] = True
    keep &= ~first
    cand = np.flatnonzero(keep)
    per_ping = len(F.HORIZONS)
    if len(cand) * per_ping > max_cases:
        cand = np.sort(rng.choice(cand, size=max(1, max_cases // per_ping), replace=False))

    # arrivals lookup: (run_key, stop_idx) -> arrive_ts
    a = arr[arr["quality"] != "coarse"]
    akey = a["run_key"].values.astype(np.int64) * 1000 + a["stop_idx"].values.astype(np.int64)
    ao = np.argsort(akey)
    akey, ats = akey[ao], a["arrive_ts"].values.astype(float)[ao]

    H = np.asarray(F.HORIZONS)
    out = {k: [] for k in ["base", "h", "dist_rem", "spread", "frac_route", "route_len", "route_speed",
                           "ts", "f_net", "e_net", "f_route", "e_route", "own_obs", "own_exp",
                           "standing_s", "actual", "hist"]}
    route_list = np.unique(rid[cand])
    for n_done, r in enumerate(route_list):
        if tick:
            tick(n_done / len(route_list))
        r = int(r)
        sel_r = cand[rid[cand] == r]
        idx_stops, ids, sp = geom[r]
        route_len = float(sp[-1])
        ev_r = route_ev.get(r, np.array([], dtype=int))
        if len(ev_r):
            ro, re_ = window_sums(minute[ev_r], obs[ev_r], exp[ev_r], minute[sel_r], F.ROUTE_WINDOW_MIN)
        else:
            ro, re_ = np.zeros(len(sel_r)), np.zeros(len(sel_r))
        no, ne = window_sums(net_m, net_o, net_e, minute[sel_r], F.NET_WINDOW_MIN)
        f_net = F.net_factor(no, ne)
        f_rt = F.route_factor(ro, re_, f_net)
        for b in np.unique(bucket[sel_r]):
            mb = bucket[sel_r] == b
            sel = sel_r[mb]
            key = (r, b)
            if key not in cums:
                cums[key] = route_cums(lk, r, ids, sp, b)
            cs, cr, csp = cums[key]
            p0 = prog[sel]
            c0s, c0r, c0p = np.interp(p0, sp, cs), np.interp(p0, sp, cr), np.interp(p0, sp, csp)
            nxt = np.searchsorted(sp, p0 + F.AHEAD_M, "right")
            T = nxt[:, None] + H[None, :] - 1
            Tc = np.minimum(T, len(sp) - 1)
            keys = run[sel][:, None].astype(np.int64) * 1000 + idx_stops[Tc]
            k = np.minimum(np.searchsorted(akey, keys), len(akey) - 1)
            actual = ats[k] - ts[sel][:, None]
            good = (T < len(sp)) & (akey[k] == keys) & (actual > 0) & (actual <= MAX_ACTUAL_S)
            ii, jj = np.nonzero(good)
            if not len(ii):
                continue
            t_ = Tc[ii, jj]
            fr = f_rt[mb][ii]
            out["base"].append(cr[t_] - c0r[ii])
            out["h"].append(H[jj])
            out["dist_rem"].append(sp[t_] - p0[ii])
            out["spread"].append(csp[t_] - c0p[ii])
            out["frac_route"].append(sp[t_] / route_len)
            out["route_len"].append(np.full(len(ii), route_len))
            out["route_speed"].append(np.full(len(ii), route_len / cr[-1] if cr[-1] > 0 else np.nan))
            out["ts"].append(ts[sel][ii])
            out["f_net"].append(f_net[mb][ii])
            out["e_net"].append(ne[mb][ii])
            out["f_route"].append(fr)
            out["e_route"].append(re_[mb][ii])
            out["own_obs"].append(own_o[sel][ii])
            out["own_exp"].append(own_e[sel][ii])
            out["standing_s"].append(standing[sel][ii])
            out["actual"].append(actual[ii, jj])
            out["hist"].append((cs[t_] - c0s[ii]) * fr)
    if tick:
        tick(1.0)
    if not out["base"]:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- training
def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def build_cases(data_dir, buckets, final_info, step, max_cases, n_folds, threads=1, log=print, bars=True):
    days = find_days(data_dir)
    if not days:
        sys.exit(f"No days with both arrivals and pings found in {data_dir} "
                 f"(pings are read from {data_dir}/arrivals/, named like 2026-10-25_pings.csv.gz)")
    arrs, pgs = [], []
    bar = Progress("reading days", len(days), bars)
    for i, (day, ap, pp) in enumerate(days):
        a, p = load_day(day, ap, pp, i * 10_000_000)
        arrs.append(a); pgs.append(p)
        bar.update()
    bar.close()
    arr, pg = pd.concat(arrs, ignore_index=True), pd.concat(pgs, ignore_index=True)
    log(f"days: {', '.join(d for d, _, _ in days)}   pings {len(pg):,}   arrivals {len(arr):,}")
    geom = route_geometry(arr)
    log(f"routes with usable geometry: {len(geom):,}")
    seg = B.load_segments(data_dir, [d for d, _, _ in days])
    if len(days) >= 2:
        group_of = {d: i for i, (d, _, _) in enumerate(days)}
        seg["grp"] = seg["day"].map(group_of).values
        arr["grp"] = arr["day"].map(group_of).values
        pg["grp"] = pg["day"].map(group_of).values
        n_groups, mode = len(days), "day"
    else:
        n_groups, mode = n_folds, "bus"
        seg["grp"] = vehicle_fold(seg["vehicle_id"], n_groups)
        arr["grp"] = vehicle_fold(arr["vehicle_id"], n_groups)
        pg["grp"] = vehicle_fold(pg["vehicle_id"], n_groups)
    workers = max(1, min(threads or 1, n_groups))
    log(f"cross-fitting by {mode}: {n_groups} groups, {workers} worker thread{'s' if workers > 1 else ''}")
    per_group = max(1, max_cases // n_groups)
    bar = Progress("building training cases", n_groups, bars)
    last = {}

    def work(g):
        def tick(frac):
            d = frac - last.get(g, 0.0)
            last[g] = frac
            if d > 0:
                bar.update(d)
        lk = make_lookup(seg[seg["grp"] != g], buckets, final_info)
        c = group_cases(pg[pg["grp"] == g], arr[arr["grp"] == g], geom, lk, buckets, step,
                        np.random.default_rng(SEED + g), per_group, tick)
        if c is not None:
            c["grp"] = np.full(len(c["base"]), g)
        tick(1.0)
        return g, c

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(work, range(n_groups)))
    else:
        results = [work(g) for g in range(n_groups)]
    bar.close()
    parts = []
    for g, c in sorted(results, key=lambda x: x[0]):
        if c is None:
            log(f"  group {g}: no cases")
        else:
            parts.append(c)
            log(f"  group {g}: {len(c['base']):,} cases")
    if not parts:
        sys.exit("No training cases could be built.")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, n_groups, mode


def fit_model(X, y, n_iter, label="training trees", bars=True, seed=SEED):
    """Grow the trees in chunks so the progress bar moves (warm start keeps the earlier trees)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=1, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=max(50, len(y) // 200), l2_regularization=1.0, early_stopping=False,
        warm_start=True, random_state=seed)
    bar = Progress(label, n_iter, bars)
    chunk = max(5, n_iter // 20)
    done = 0
    while done < n_iter:
        nxt = min(n_iter, done + chunk)
        m.set_params(max_iter=nxt)
        m.fit(X, y)
        bar.update(nxt - done)
        done = nxt
    bar.close()
    return m


def pick_iterations(X, y, hold, max_iter, bars=True):
    """Number of trees that is best on the held-out group (0 = the model does not help)."""
    m = fit_model(X[~hold], y[~hold], max_iter, "fitting (held-out check)", bars)
    errs = [float(np.mean(np.abs(y[hold])))]
    bar = Progress("scoring held-out cases", max_iter, bars)
    preds = []
    for p in m.staged_predict(X[hold]):
        errs.append(float(np.mean(np.abs(y[hold] - p))))
        preds.append(p)
        bar.update()
    bar.close()
    best = int(np.argmin(errs))
    pred = preds[best - 1] if best else np.zeros(int(hold.sum()))
    return best, errs[0], errs[best], pred


def report(c, pred_resid):
    actual, hist = c["actual"], c["hist"]
    ai = np.maximum(c["hist"] + pred_resid, 0.0)
    lines = [f"{'h':>3} {'cases':>9} {'history':>9} {'history+live':>13} {'model':>9}   (MAE seconds, held-out)"]
    raw = c["base"]
    for h in (None,) + F.HORIZONS:
        m = np.ones(len(actual), bool) if h is None else c["h"] == h
        if m.sum() == 0:
            continue
        lines.append(f"{'all' if h is None else h:>3} {m.sum():>9,} {mae(actual[m], raw[m]):>9.1f} "
                     f"{mae(actual[m], hist[m]):>13.1f} {mae(actual[m], ai[m]):>9.1f}")
    return "\n".join(lines), mae(actual, hist), mae(actual, ai), mae(actual, raw)


def cmd_train(a, fast=False, log=print, bars=True):
    info_path = Path(a.data) / "eta_bucketed_info.json"
    if not info_path.exists():
        sys.exit(f"{info_path} not found: run  python build/build_eta_bucketed.py build  first")
    final_info = json.load(open(info_path))
    buckets = [tuple(b) for b in final_info["buckets"]]
    if final_info.get("split_daytype"):
        sys.exit("train_eta does not support a table built with --split-daytype")
    lim = limit_threads(a.threads)
    c, n_groups, mode = build_cases(a.data, buckets, final_info, a.step, a.max_cases, a.folds,
                                    a.threads, log, bars)
    X = F.build_matrix(c)
    y = c["actual"] - c["hist"]
    hold = c["grp"] == (n_groups - 1)
    if hold.sum() < 200 or (~hold).sum() < 200:
        hold = np.zeros(len(y), bool)
        hold[::5] = True
    log(f"\ncases {len(y):,}   features {X.shape[1]}   held-out group {n_groups - 1} ({hold.sum():,} cases)")
    t0 = time.time()
    best, e0, e1, pred = pick_iterations(X, y, hold, 120 if fast else 300, bars)
    held = {k: v[hold] for k, v in c.items()}
    text, mh, ma, mr = report(held, pred)
    log(text)
    log(f"\nheld-out MAE: table only {mr:.1f}s, table x live ratio {mh:.1f}s, model {ma:.1f}s  "
        f"({best} trees chosen, {time.time() - t0:.0f}s)")
    final = fit_model(X, y, max(best, 1), "fitting final model", bars)
    out = Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": int(len(y)), "groups": mode,
        "days": final_info.get("days", []), "horizons": list(F.HORIZONS),
        "heldout_mae_table_s": round(mr, 1), "heldout_mae_history_live_s": round(mh, 1),
        "heldout_mae_model_s": round(ma, 1),
        "better_than_history": bool(best > 0 and ma < mh * 0.99), "trees": int(best),
    }
    import joblib
    joblib.dump({"model": final, "features": F.FEATURES, "info": info}, out / "eta_model.joblib")
    json.dump(info, open(out / "eta_model_info.json", "w"), indent=1)
    log(f"\nwrote {out / 'eta_model.joblib'}")
    if not info["better_than_history"]:
        log("NOTE: the model is not better than table x live ratio on held-out data (normal with few days). "
            "The planner will ignore it and keep using table x live ratio.")
    return ma, mh


def cmd_check(a):
    p = Path(a.models) / "eta_model_info.json"
    if not p.exists():
        sys.exit(f"{p} not found: run  python build/train_eta.py train")
    print(json.dumps(json.load(open(p)), indent=1))


# ---------------------------------------------------------------- selftest
def _synthetic(root, days=("2026-11-01", "2026-11-02", "2026-11-03")):
    """Write fake arrivals/pings for one 40-stop route with slow spells shared by all buses."""
    rng = np.random.default_rng(3)
    root = Path(root)
    (root / "arrivals").mkdir(parents=True, exist_ok=True)
    (root / "pings").mkdir(parents=True, exist_ok=True)
    n_st = 40
    sp = np.arange(n_st) * 500.0 + 300
    sids = np.arange(n_st) + 100
    for di, day in enumerate(days):
        t0 = pd.Timestamp(day, tz="Asia/Kolkata").timestamp() + 9 * 3600
        slow = rng.uniform(0.8, 1.7, size=40)                   # one value per 15 minutes
        arr_rows, ping_rows = [], []
        for run in range(60):
            start = t0 + run * 150 + rng.uniform(0, 60)
            t, pos, k = start, 0.0, 0
            veh = f"DL1P{run % 12:04d}"
            arrived = {}
            while pos < sp[-1] + 100:
                f = slow[min(int((t - t0) // 900), 39)] * rng.lognormal(0, 0.15)
                step_t = 30.0
                spd = 8.0 / f
                pos += spd * step_t
                t += step_t
                ping_rows.append((run + di * 1000, 7, veh, int(t), pos))
            # arrival times by linear interpolation of the pings of this run
            pr = [r for r in ping_rows if r[0] == run + di * 1000]
            tt = np.array([r[3] for r in pr], float); pp = np.array([r[4] for r in pr], float)
            for i in range(n_st):
                at = float(np.interp(sp[i], pp, tt))
                arr_rows.append((run + di * 1000, 7, veh, "x", i, int(sids[i]), at, "ok", sp[i]))
        pd.DataFrame(arr_rows, columns=["run_id", "route_id", "vehicle_id", "trip_id", "stop_idx", "stop_id",
                                        "arrive_ts", "quality", "stop_prog_m"]).to_csv(
            root / "arrivals" / f"{day}_arrivals.csv.gz", index=False)
        pd.DataFrame(ping_rows, columns=["run_id", "route_id", "vehicle_id", "ts", "prog"]).to_csv(
            root / "pings" / f"{day}_pings.csv.gz", index=False)


def cmd_selftest(a):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        _synthetic(root)
        seg = B.load_segments(str(root))
        buckets = B.parse_buckets(B.DEFAULT_BUCKETS)
        sb, sall, _ = B.fit_scale(seg, buckets, False)
        table, info, _ = B.build_tables(seg, buckets, False)
        info["scale_bucket"], info["scale_all"] = sb, sall
        table.to_csv(root / "eta_bucketed.csv.gz", index=False)
        json.dump(info, open(root / "eta_bucketed_info.json", "w"))
        ns = argparse.Namespace(data=str(root), models=str(Path(tmp) / "models"), step=2, max_cases=200000, folds=3,
                                threads=2)
        ma, mh = cmd_train(ns, fast=True, log=lambda *x: None, bars=False)
        assert (Path(tmp) / "models" / "eta_model.joblib").exists()
        print(f"selftest: held-out MAE history x live {mh:.1f}s, model {ma:.1f}s")
        assert ma <= mh * 1.001, "the chosen model must not be worse than history x live ratio on held-out data"
        print("PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--models", default="models")
    t.add_argument("--step", type=int, default=3, help="use every Nth ping as a case start")
    t.add_argument("--max-cases", type=int, default=2_000_000)
    t.add_argument("--threads", type=int, default=0,
                   help="CPU threads to use (0 = all). Lower it to keep the machine usable, e.g. --threads 2")
    t.add_argument("--folds", type=int, default=3, help="bus groups when there is only one day")
    c = sub.add_parser("check"); c.add_argument("--models", default="models")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"train": cmd_train, "check": cmd_check, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
