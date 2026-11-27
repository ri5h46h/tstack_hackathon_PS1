"""
search.py  -  journey search on top of Network and the bucketed ETA table.

Method: RAPTOR-style rounds. Round k finds the earliest arrival at every stop
using at most k bus rides. Waiting time at each boarding is half the headway
of the route at that hour (headway comes from the GTFS timetable). Ride times
come from the learned ETA table (EtaLookup), so they depend on the time of day.
Walking links join nearby stops for transfers.

Usage
    p = Planner("gtfs", "data")
    result = p.plan((28.70, 77.10), (28.55, 77.25), "09:30")
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT))

from build_eta_bucketed import EtaLookup  # noqa: E402
from planner.network import Network, WALK_DETOUR, WALK_SPEED  # noqa: E402

INF = 1e18
IST_OFFSET = 19800
MIN_GAIN_S = 60  # an option with more transfers must save at least this much


def _nearby(net, lat, lon, max_m=1000.0, k=10, min_k=3):
    """Stops a person can walk to. Returns {stop_index: walking seconds}."""
    d = net.dist_to(lat, lon)
    order = np.argsort(d)[:k]
    sel = [int(i) for i in order if d[i] <= max_m]
    if not sel:  # nothing close: take the nearest few so the rider still gets an answer
        sel = [int(i) for i in order[:min_k]]
    return {i: float(d[i] * WALK_DETOUR / WALK_SPEED) for i in sel}


def _raptor(net, ride, avail, waitfn, access, egress, max_rounds):
    n = len(net.stop_ids)
    tau = [np.full(n, INF)]
    for s, t in access.items():
        tau[0][s] = t
    parent = [{}]
    marked = set(access)
    options = []
    best_total = INF
    for k in range(1, max_rounds + 1):
        prev = tau[k - 1]
        new = prev.copy()
        par = {}
        q = {}
        for s in marked:
            for r, pos in net.stop_routes[s]:
                if not avail[r]:
                    continue
                if r not in q or pos < q[r]:
                    q[r] = pos
        improved = set()
        for r, p0 in q.items():
            pat = net.patterns[r]
            times = ride[r]
            cur = INF
            bpos = -1
            for i in range(p0, len(pat)):
                s = pat[i]
                if cur < INF:
                    cur += times[i - 1]
                    if cur < new[s]:
                        new[s] = cur
                        par[s] = ("ride", r, bpos, i)
                        improved.add(s)
                if prev[s] >= INF:
                    continue
                a = prev[s] + waitfn(r, i, prev[s])
                if a < cur:
                    cur = a
                    bpos = i
        walked = set()
        for s in list(improved):
            ts = new[s]
            for t, wsec in net.foot[s]:
                if ts + wsec < new[t]:
                    new[t] = ts + wsec
                    par[t] = ("walk", s)
                    walked.add(t)
        marked = improved | walked
        tau.append(new)
        parent.append(par)
        if not marked:
            break
        tot, s_end = INF, -1
        for s, e in egress.items():
            if new[s] + e < tot:
                tot, s_end = new[s] + e, s
        if s_end >= 0 and tot < best_total - MIN_GAIN_S:
            options.append((k, s_end, tot))
            best_total = tot
    return options, parent


def _legs(net, parent, k, s_end):
    """Walk the parent pointers back to the origin. Returns legs in travel order."""
    legs, s, kk = [], s_end, k
    guard = 0
    while kk > 0 and guard < 1000:
        guard += 1
        p = parent[kk].get(s)
        if p is None:
            kk -= 1
            continue
        if p[0] == "walk":
            legs.append(("walk", p[1], s))
            s = p[1]
        else:
            _, r, bp, ap = p
            legs.append(("ride", r, bp, ap))
            s = int(net.patterns[r][bp])
            kk -= 1
    legs.append(("access", s))
    legs.reverse()
    return legs


def _tidy_walks(desc):
    """Join walking legs that follow each other, drop zero-minute walks, relabel."""
    out = []
    for d in desc:
        if d["type"] == "walk" and out and out[-1]["type"] == "walk":
            p = out[-1]
            p["minutes"] = round(p["minutes"] + d["minutes"], 1)
            if "to" in d:
                p["to"] = d["to"]
            p["path"] = p["path"] + d["path"][1:] if d["path"] and p["path"] and d["path"][0] == p["path"][-1] \
                else p["path"] + d["path"]
        else:
            out.append(dict(d))
    out = [d for d in out if not (d["type"] == "walk" and d["minutes"] < 0.1)]
    for i, d in enumerate(out):
        if d["type"] == "walk":
            d["what"] = ("walk to first stop" if i == 0 else
                         "walk to destination" if i == len(out) - 1 else "walk to transfer stop")
    return out


def _fmt_clock(sec):
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}"


class Planner:
    def __init__(self, gtfs_dir="gtfs", data_dir="data", rebuild=False):
        t0 = time.time()
        self.net = Network(gtfs_dir, data_dir, rebuild=rebuild)
        d = Path(data_dir)
        self.lk = EtaLookup(d / "eta_bucketed.csv.gz", d / "eta_bucketed_info.json")
        self._ride_cache = {}
        self.live = None
        self.load_s = time.time() - t0

    def attach_live(self, live_model):
        """Give the planner a LiveModel (planner/live.py). Used for 'now' trips."""
        self.live = live_model

    # -------------------------------------------------------- ride times
    def _edge_stats(self, bucket):
        """Per route, per edge between consecutive stops, for one time bucket:
        scaled seconds, raw (unscaled) seconds, spread (p75-p25, scaled) and a real-data flag."""
        if bucket in self._ride_cache:
            return self._ride_cache[bucket]
        net, lk = self.net, self.lk
        out = {"scaled": [], "raw": [], "spread": [], "real": []}
        sc = lk.scale(bucket)
        for r, pat in enumerate(net.patterns):
            rid = int(net.route_ids[r])
            m = max(0, len(pat) - 1)
            t, w, sp, f = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m, dtype=bool)
            for i in range(m):
                sec, spr, src = lk.seg_stats(rid, int(net.stop_ids[pat[i]]), int(net.stop_ids[pat[i + 1]]),
                                             bucket, float(net.seg_dist[r][i]), scaled=False)
                if not sec:
                    sec, spr = 60.0 / sc if sc else 60.0, 30.0
                w[i], t[i], sp[i] = sec, sec * sc, spr * sc
                f[i] = src in ("pair_blend", "pair_all")
            out["scaled"].append(t); out["raw"].append(w); out["spread"].append(sp); out["real"].append(f)
        self._ride_cache[bucket] = out
        return out

    def _ride_times(self, bucket):
        st = self._edge_stats(bucket)
        return st["scaled"], st["real"]

    # -------------------------------------------------------- planning
    def plan(self, origin, dest, depart="now", max_transfers=2, speed_factor=1.0,
             max_options=3, walk_radius_m=1000.0, use_live=True):
        """
        origin, dest: (lat, lon). depart: 'HH:MM' (IST) or 'now'.
        speed_factor: extra multiplier on all ride times (1.0 = none).
        use_live: for 'now' trips, scale ride times by the live traffic factor and
                  list buses approaching each boarding stop.
        """
        net = self.net
        is_now = depart == "now"
        if is_now:
            sec = (time.time() + IST_OFFSET) % 86400
        else:
            hh, mm = depart.split(":")[:2]
            sec = int(hh) * 3600 + int(mm) * 60
        hour = int(sec // 3600) % 24
        bucket = self.lk.bucket_for(hour)
        times, real = self._ride_times(bucket)

        fnet, rfac = 1.0, {}
        live_info = {"applied": False}
        if use_live and is_now and self.live is not None:
            f = self.live.factors()
            if f["warm"]:
                fnet, rfac = f["net"], f["routes"]
                live_info = {"applied": True, "network_factor": round(fnet, 3), "events": f["n_events"]}
            else:
                live_info = {"applied": False, "reason": "live data is still warming up"}
        elif use_live and is_now:
            live_info = {"applied": False, "reason": "live feed is not running"}
        elif use_live:
            live_info = {"applied": False, "reason": "live data is used only for 'now' trips"}
        if fnet == 1.0 and not rfac and speed_factor == 1.0:
            ride = times
        else:
            ride = [t * (rfac.get(r, fnet) * speed_factor) for r, t in enumerate(times)]
        waits = net.waits(hour)
        avail = [w is not None for w in waits]
        wm = None
        now_ts = time.time()
        if is_now and use_live and self.live is not None:
            wm = self.live.wait_model(now_ts, waits)
            avail = [a or wm.has_route(r) for r, a in enumerate(avail)]

        def waitfn(r, i, arrive):
            if wm is not None:
                w = wm.wait_info(r, i, arrive)[0]
                if w is not None:
                    return w
            return waits[r] if waits[r] is not None else 900.0

        access = _nearby(net, origin[0], origin[1], walk_radius_m)
        egress = _nearby(net, dest[0], dest[1], walk_radius_m)
        options, parent = _raptor(net, ride, avail, waitfn, access, egress, max_transfers + 1)

        out = []
        for k, s_end, total in options[:max_options]:
            legs = _legs(net, parent, k, s_end)
            opt = self._describe(legs, access, egress, s_end, ride, real, waits, sec, origin, dest, wm,
                                  now_ts if wm is not None else None)
            if wm is not None:
                for leg in opt["legs"]:
                    if leg["type"] == "bus":
                        ri, si = net.route_index(leg["route_id"]), net.stop_index(leg["from_stop_id"])
                        leg["live_buses"] = self.live.approaching(si, ri, max_n=3) if None not in (ri, si) else []
            out.append(opt)
        out.sort(key=lambda o: o["total_min"])
        return {
            "depart": _fmt_clock(sec),
            "bucket": bucket,
            "speed_factor": speed_factor,
            "live": live_info,
            "origin_stops_nearby": len(access),
            "dest_stops_nearby": len(egress),
            "options": out,
        }

    def _describe(self, legs, access, egress, s_end, ride, real, waits, start_sec, origin, dest,
                  wm=None, now_ts=None):
        net = self.net

        def pt(i):
            return [float(net.stop_lat[i]), float(net.stop_lon[i])]

        t = 0.0
        first = legs[0][1]
        t += access[first]
        desc = [{"type": "walk", "to": net.stop_names[first], "minutes": round(access[first] / 60, 1),
                 "what": "walk to first stop", "path": [[float(origin[0]), float(origin[1])], pt(first)]}]
        n_real = n_all = 0
        rd_used = 0.0
        by_bucket = {b[0]: 0.0 for b in self.lk.buckets}
        for leg in legs[1:]:
            if leg[0] == "walk":
                sec = net.walk_seconds(leg[1], leg[2])
                t += sec
                desc.append({"type": "walk", "from": net.stop_names[leg[1]],
                             "to": net.stop_names[leg[2]], "minutes": round(sec / 60, 1),
                             "what": "walk to transfer stop", "path": [pt(leg[1]), pt(leg[2])]})
            else:
                _, r, bp, ap = leg
                w, veh, wsrc, esrc = self._wait(r, bp, t, waits, wm, now_ts)
                rd = float(ride[r][bp:ap].sum())
                rd_used += rd
                for bn in by_bucket:
                    by_bucket[bn] += float(self._edge_stats(bn)["scaled"][r][bp:ap].sum())
                t += w + rd
                n_real += int(real[r][bp:ap].sum())
                n_all += ap - bp
                pat = net.patterns[r]
                desc.append({
                    "type": "bus", "route": net.route_short[r], "route_id": int(net.route_ids[r]),
                    "description": net.route_desc[r],
                    "from": net.stop_names[int(pat[bp])], "from_stop_id": int(net.stop_ids[pat[bp]]),
                    "to": net.stop_names[int(pat[ap])], "to_stop_id": int(net.stop_ids[pat[ap]]),
                    "stops": int(ap - bp), "wait_min": round(w / 60, 1),
                    "wait_source": wsrc, "next_vehicle": veh, "eta_source": esrc,
                    "ride_min": round(rd / 60, 1),
                    "path": net.route_path(r, bp, ap),
                })
        last = egress[s_end]
        t += last
        desc.append({"type": "walk", "from": net.stop_names[s_end], "minutes": round(last / 60, 1),
                     "what": "walk to destination", "path": [pt(s_end), [float(dest[0]), float(dest[1])]]})
        desc = _tidy_walks(desc)
        share = n_real / n_all if n_all else 0.0
        conf = "high" if share >= 0.8 else "medium" if share >= 0.5 else "low"
        buses = sum(1 for d in desc if d["type"] == "bus")
        quiet = {}
        if by_bucket and any(v > 0 for v in by_bucket.values()):
            qb = min((b for b in by_bucket if by_bucket[b] > 0), key=lambda b: by_bucket[b])
            rng = next(x for x in self.lk.buckets if x[0] == qb)
            q = (t - rd_used + by_bucket[qb]) / 60
            quiet = {"quiet_hour_min": round(min(q, t / 60), 1),
                     "quiet_hour_bucket": f"{qb.replace('_', ' ')} ({rng[1]}-{rng[2]})",
                     "slower_than_quiet_min": round(max(0.0, t / 60 - q), 1)}
        return {
            **quiet,
            "total_min": round(t / 60, 1),
            "arrive": _fmt_clock(start_sec + t),
            "transfers": max(0, buses - 1),
            "confidence": conf,
            "data_share": round(share, 2),
            "legs": desc,
        }

    def _wait(self, r, bp, arrive_s, waits, wm, now_ts):
        """Wait at boarding position bp for a rider arriving arrive_s seconds from now.
        Returns (seconds, vehicle name, 'live'|'timetable', 'model'|'history'|None)."""
        base = waits[r]
        if wm is not None:
            w, veh, src = wm.wait_info(r, bp, arrive_s)
            if src == "live":
                esrc = "history"
                net = self.net
                si = int(net.patterns[r][bp])
                try:
                    cand = self.live.approaching(si, r, now=now_ts, max_n=6)
                except Exception:  # noqa: BLE001
                    cand = []
                for c in cand:
                    if c["eta_s"] >= arrive_s + 15:
                        w2 = c["eta_s"] - arrive_s
                        if base is None or w2 <= 3 * base:
                            return float(w2), c["vehicle"], "live", c["eta_source"]
                        break
                return float(w), veh, "live", esrc
            if w is not None:
                return float(w), None, "timetable", None
        return float(base if base is not None else 900.0), None, "timetable", None
