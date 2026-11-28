"""
live.py  -  live bus data for the planner.

Four parts:

1. LiveFeed   reads the GTFS-realtime VehiclePositions feed (Transport Stack, or OTD as
              backup) every ~15 s in a background thread. Keys come from the environment
              (.env): TSTACK_API_KEY and OTD_API_KEY.

2. LiveModel  follows each bus along the road shape of its route and compares what buses
              are doing now with what the ETA table says is usual for this time of day
              (the "ratio", see planner/features.py). It lists the buses approaching a
              stop with an ETA each.

3. ETA        history ETA = table time from the bus's position to the stop x live ratio.
              If a trained model exists (models/eta_model.joblib, made by build/train_eta.py)
              it corrects the history ETA using the live ratios and the bus's own recent
              behaviour. Without a model the history ETA is used.
              In both cases: ETA = remaining time - seconds since the bus last reported.

4. WaitModel  for the journey search: how long until the next live bus reaches a stop,
              given the time the rider gets there.
"""
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from planner import features as F

TSTACK_URL = "https://dts-backend.transportstack.in/api/dataset/otd/get-file"
TSTACK_PARAMS = {"agency": "delhi-buses", "category": "realtime_gtfs", "filename": "VehiclePositions.pb"}
OTD_URL = "https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb"
BBOX = (28.20, 29.10, 76.60, 77.70)     # lat min, lat max, lon min, lon max

MAX_PING_AGE_S = 600       # drop pings older than this
FUTURE_SLACK_S = 120
ON_ROUTE_M = 150           # a bus farther than this from its route's road shape is ignored
PAST_STOP_M = 40           # a bus this far past a stop no longer counts as approaching it
STALE_S = 300
BOARD_MARGIN_S = 15        # a rider needs this long to be at the stop before the bus arrives


# ------------------------------------------------------------------ .env
def load_env(path):
    """Read KEY=VALUE lines into os.environ (does not overwrite existing values)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip().removeprefix("export ").strip(), v)


# ------------------------------------------------------------------ decoding
def decode_vehicle_positions(content, source, now=None):
    """Bytes of VehiclePositions.pb -> list of dicts, one per bus. Bad rows are dropped."""
    from google.transit import gtfs_realtime_pb2  # package: gtfs-realtime-bindings

    now = now or time.time()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(content)
    rows = []
    for e in feed.entity:
        if not e.HasField("vehicle"):
            continue
        v = e.vehicle
        if not v.HasField("position"):
            continue
        lat, lon = float(v.position.latitude), float(v.position.longitude)
        if not (BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]):
            continue
        ts = int(v.timestamp) if v.timestamp else int(feed.header.timestamp or 0)
        if ts <= 0 or ts > now + FUTURE_SLACK_S or now - ts > MAX_PING_AGE_S:
            continue
        label = getattr(v.vehicle, "label", "") or ""
        plate = getattr(v.vehicle, "license_plate", "") or ""
        vid = v.vehicle.id or label or e.id
        rid = v.trip.route_id
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            rid = None                       # no usable route id: counted as an unresolved bus
        rows.append({
            "vehicle_id": str(vid), "name": str(plate or vid), "label": str(label),
            "route_id": rid, "trip_id": v.trip.trip_id,
            "lat": lat, "lon": lon, "ts": ts,
            # only Transport Stack sends a speed; its unit is unverified, so it is shown, not used
            "speed_rep": float(v.position.speed) if source == "tstack" and v.position.HasField("speed") else None,
        })
    return rows


# ------------------------------------------------------------------ the feed
class LiveFeed:
    def __init__(self, tstack_key=None, otd_key=None, primary="tstack", interval=15, timeout=25):
        self.keys = {"tstack": tstack_key, "otd": otd_key}
        self.primary = primary if self.keys.get(primary) else next((s for s, k in self.keys.items() if k), None)
        self.interval = max(10, interval)
        self.timeout = timeout
        self.active = self.primary
        self.fails = 0
        self.cycle = 0
        self.rows = []
        self.fetched_at = 0.0
        self.source = None
        self.polls = self.ok_polls = self.rejected = 0
        self.last_error = None
        self.callbacks = []
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env_path=None, **kw):
        if env_path:
            load_env(env_path)
        return cls(os.environ.get("TSTACK_API_KEY"), os.environ.get("OTD_API_KEY"), **kw)

    @property
    def available(self):
        return self.primary is not None

    def add_callback(self, fn):
        self.callbacks.append(fn)

    # -- network
    def _get(self, source):
        import requests
        if source == "tstack":
            r = requests.get(TSTACK_URL, params=TSTACK_PARAMS, headers={"x-api-key": self.keys["tstack"]},
                             timeout=self.timeout)
        else:
            r = requests.get(OTD_URL, params={"key": self.keys["otd"]}, timeout=self.timeout)
        if r.status_code in (401, 403):
            raise RuntimeError(f"{source}: key rejected (HTTP {r.status_code})")
        if r.status_code == 429:
            raise RuntimeError(f"{source}: rate limited (HTTP 429)")
        r.raise_for_status()
        body = r.content
        if body[:1] in (b"{", b"<", b"["):
            raise RuntimeError(f"{source}: got text, not a feed")
        return body

    def refresh_once(self):
        """One poll. Returns True if a good snapshot was stored."""
        if not self.available:
            return False
        self.cycle += 1
        src = self.active
        other = "otd" if src == "tstack" else "tstack"
        if src != self.primary and self.cycle % 4 == 0:
            src = self.primary            # on the backup: try the main feed every 4th poll
        self.polls += 1
        try:
            rows = decode_vehicle_positions(self._get(src), src)
            prev = len(self.rows)
            if not rows or (prev >= 200 and len(rows) < 0.2 * prev):
                self.rejected += 1
                raise RuntimeError(f"{src}: snapshot rejected ({len(rows)} buses, was {prev})")
        except Exception as e:  # noqa: BLE001  (any failure counts as a failed poll)
            self.last_error = str(e)
            self.fails += 1
            if self.fails >= 2 and self.keys.get(other):
                self.active, self.fails = other, 0
            return False
        now = time.time()
        with self._lock:
            self.rows, self.fetched_at, self.source = rows, now, src
        self.active, self.fails, self.last_error = src, 0, None
        self.ok_polls += 1
        for cb in self.callbacks:
            try:
                cb(rows, now, src)
            except Exception as e:  # noqa: BLE001
                self.last_error = f"callback: {e}"
        return True

    # -- thread
    def start(self):
        if not self.available or self._thread:
            return
        def loop():
            while not self._stop.is_set():
                t0 = time.time()
                self.refresh_once()
                wait = self.interval if self.fails == 0 else 5
                self._stop.wait(max(1.0, wait - (time.time() - t0)))
        self._thread = threading.Thread(target=loop, daemon=True, name="live-feed")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def status(self):
        age = time.time() - self.fetched_at if self.fetched_at else None
        return {
            "available": self.available, "source": self.source, "active": self.active,
            "buses_in_feed": len(self.rows), "age_s": round(age, 1) if age is not None else None,
            "polls": self.polls, "ok_polls": self.ok_polls, "rejected": self.rejected,
            "last_error": self.last_error,
        }


# ------------------------------------------------------------------ route geometry for live use
class _Geom:
    """A route's road shape (or straight lines between stops) with distance along it."""

    def __init__(self, net, r):
        x, y = net.geom_x[r], net.geom_y[r]
        self.ax, self.ay = x[:-1], y[:-1]
        self.vx, self.vy = x[1:] - self.ax, y[1:] - self.ay
        self.l2 = np.maximum(self.vx ** 2 + self.vy ** 2, 1e-6)
        self.seglen = np.sqrt(self.l2)
        self.cum = net.geom_cum[r]
        self.stop_prog = net.stop_prog[r]
        self.total = float(self.cum[-1])

    def project(self, x, y, near=None):
        """(distance from the line, progress along it). near=(lo, hi) prefers that stretch of the route."""
        t = np.clip(((x - self.ax) * self.vx + (y - self.ay) * self.vy) / self.l2, 0.0, 1.0)
        d = np.hypot(x - (self.ax + t * self.vx), y - (self.ay + t * self.vy))
        i = int(np.argmin(d))
        if near is not None:
            m = (self.cum[1:] >= near[0]) & (self.cum[:-1] <= near[1])
            if m.any():
                dw = np.where(m, d, np.inf)
                j = int(np.argmin(dw))
                if dw[j] <= ON_ROUTE_M:
                    i = j
        return float(d[i]), float(self.cum[i] + t[i] * self.seglen[i])


# ------------------------------------------------------------------ the live model
class LiveModel:
    def __init__(self, planner, eta_model_path=None):
        self.planner = planner
        self.net = planner.net
        self.lk = planner.lk
        self.lock = threading.RLock()
        self._geoms = {}
        self._cum = {}                           # (route, bucket, kind) -> cumulative time at each stop
        self.buses = defaultdict(dict)           # route index -> {vehicle id: state}
        self.net_bins = deque()                  # [minute, sum observed s, sum expected s, n]
        self.route_bins = defaultdict(deque)
        self.updates = 0
        self.last_update = 0.0
        self.counts = defaultdict(int)
        self.live_total = 0                      # every bus in the latest snapshot, resolved or not
        self.unresolved = {}                     # vehicle id -> route id, buses of the latest snapshot
        self._fcache = (0.0, None)
        self.eta_model = None
        self.eta_info = {"loaded": False, "reason": "no model file"}
        if eta_model_path:
            self.load_eta_model(eta_model_path)

    # -- ETA model
    def load_eta_model(self, path):
        p = Path(path)
        if not p.exists():
            self.eta_info = {"loaded": False, "reason": f"{p.name} not found (run build/train_eta.py train)"}
            return False
        try:
            import joblib
            n = int(os.environ.get("ETA_THREADS", "0") or 0)
            if n > 0:                      # keep the live model's CPU use small (set ETA_THREADS in .env)
                from threadpoolctl import threadpool_limits
                threadpool_limits(limits=n)
            obj = joblib.load(p)
            if obj.get("features") != F.FEATURES:
                self.eta_info = {"loaded": False, "reason": "model was trained with different features; retrain"}
                return False
            if not obj.get("info", {}).get("better_than_history", False):
                self.eta_info = {"loaded": False, **obj.get("info", {}),
                                 "reason": "model was not better than table x live ratio on held-out data; not used"}
                return False
            self.eta_model = obj["model"]
            self.eta_info = {"loaded": True, **obj.get("info", {})}
            return True
        except Exception as e:  # noqa: BLE001  (a bad model file must never stop the planner)
            self.eta_info = {"loaded": False, "reason": f"could not load model: {e}"}
            return False

    # -- helpers
    def geom(self, r):
        g = self._geoms.get(r)
        if g is None:
            g = self._geoms[r] = _Geom(self.net, r)
        return g

    def cum_time(self, r, bucket, kind="scaled"):
        """Cumulative table time (s) at each stop of route r. kind: scaled | raw | spread."""
        key = (r, bucket, kind)
        c = self._cum.get(key)
        if c is None:
            st = self.planner._edge_stats(bucket)
            c = self._cum[key] = np.concatenate([[0.0], np.cumsum(st[kind][r])])
        return c

    def _bucket(self, ts):
        return self.lk.bucket_for(int(((ts + 19800) // 3600) % 24))

    @staticmethod
    def _add_bin(dq, minute, obs, exp):
        if dq and dq[-1][0] == minute:
            dq[-1][1] += obs
            dq[-1][2] += exp
            dq[-1][3] += 1
        else:
            dq.append([minute, obs, exp, 1])

    # -- feeding
    def update(self, rows, now=None, source=None):
        now = now or time.time()
        with self.lock:
            unresolved = {}
            for row in rows:
                r = self.net.route_index(row["route_id"]) if row["route_id"] is not None else None
                if r is None:
                    unresolved[row["vehicle_id"]] = row["route_id"]   # route id missing from the static GTFS
                    self.counts["unknown_route"] += 1
                    continue
                if len(self.net.patterns[r]) < 2:
                    self.counts["route_without_stops"] += 1
                    continue
                x, y = self.net.xy(row["lat"], row["lon"])
                g = self.geom(r)
                vid, ts = row["vehicle_id"], row["ts"]
                st = self.buses[r].get(vid)
                near = None
                if st is not None and ts > st["ts"]:
                    dt = ts - st["ts"]
                    near = (st["prog"] - 200, st["prog"] + F.VMAX * max(dt, 30) + 400)
                perp, prog = g.project(x, y, near)
                if perp > ON_ROUTE_M:
                    self.counts["off_route"] += 1
                    self.buses[r].pop(vid, None)
                    continue
                if st is None or ts > st["ts"]:
                    hist = deque(st["hist"], maxlen=F.OWN_K) if st is not None else deque(maxlen=F.OWN_K)
                    if st is not None:
                        hist.append(self._interval(r, g, st, ts, prog))
                    else:
                        hist.append((0.0, 0.0))
                    moved = st is None or abs(prog - st["move_prog"]) >= F.MOVE_M
                    self.buses[r][vid] = {
                        "ts": ts, "prog": prog, "perp": perp, "lat": row["lat"], "lon": row["lon"],
                        "name": row.get("name") or vid, "label": row.get("label", ""),
                        "speed_rep": row.get("speed_rep"), "hist": hist,
                        "move_prog": prog if moved else st["move_prog"],
                        "move_ts": ts if moved else st["move_ts"],
                    }
            for r in list(self.buses):
                d = self.buses[r]
                for v in [v for v, s in d.items() if now - s["ts"] > MAX_PING_AGE_S]:
                    del d[v]
                if not d:
                    del self.buses[r]
            lim = int(now // 60) - F.ROUTE_WINDOW_MIN - 2
            while self.net_bins and self.net_bins[0][0] < lim:
                self.net_bins.popleft()
            for dq in self.route_bins.values():
                while dq and dq[0][0] < lim:
                    dq.popleft()
            self.unresolved = unresolved
            self.live_total = len({row["vehicle_id"] for row in rows})
            self.updates += 1
            self.last_update = now

    def _interval(self, r, g, st, ts, prog):
        """Record one ping interval. Returns (observed s, expected s) for the bus's own history."""
        dt = ts - st["ts"]
        p0 = st["prog"]
        sp = g.stop_prog
        if not bool(F.interval_ok(dt, p0, prog, sp[-1])):
            self.counts["skipped_interval"] += 1
            return (0.0, 0.0)
        ct = self.cum_time(r, self._bucket(ts), "scaled")
        exp = max(0.0, float(np.interp(prog, sp, ct) - np.interp(p0, sp, ct)))
        minute = int(ts // 60)
        self._add_bin(self.net_bins, minute, dt, exp)
        self._add_bin(self.route_bins[r], minute, dt, exp)
        return (float(dt), exp)

    # -- ratios
    @staticmethod
    def _sum(dq, lo, hi):
        o = e = n = 0.0
        for m, so, se, c in dq:
            if lo <= m <= hi:
                o, e, n = o + so, e + se, n + c
        return o, e, int(n)

    def factors(self, now=None):
        """
        dict: net (ratio), warm, routes {route index: ratio}, n_events, expected_s,
              raw_net (obs, exp), raw_routes {route index: (obs, exp)}
        The current partial minute is left out, so training and serving see the same windows.
        """
        now = now or time.time()
        t0, cached = self._fcache
        if cached is not None and abs(now - t0) < 3:
            return cached
        with self.lock:
            m = int(now // 60)
            o, e, n = self._sum(self.net_bins, m - F.NET_WINDOW_MIN, m - 1)
            fnet = float(F.net_factor(o, e))
            warm = e >= F.WARM_EXPECTED_S
            raw, routes = {}, {}
            for r, dq in self.route_bins.items():
                ro, re_, _ = self._sum(dq, m - F.ROUTE_WINDOW_MIN, m - 1)
                raw[r] = (ro, re_)
                if warm and re_ > 0:
                    routes[r] = float(F.route_factor(ro, re_, fnet))
            out = {"net": fnet, "warm": warm, "routes": routes, "n_events": n, "expected_s": round(e),
                   "raw_net": (o, e), "raw_routes": raw}
        self._fcache = (now, out)
        return out

    def route_factor(self, r, f):
        return f["routes"].get(r, f["net"])

    # -- buses approaching a stop
    def approaching(self, stop_idx, route_idx=None, now=None, max_n=4, max_eta_s=5400, use_ai=True):
        now = now or time.time()
        f = self.factors(now)
        cand = []
        with self.lock:
            for r, pos in self.net.stop_routes[stop_idx]:
                if route_idx is not None and r != route_idx:
                    continue
                buses = self.buses.get(r)
                if not buses:
                    continue
                g = self.geom(r)
                sp = g.stop_prog
                if pos >= len(sp):
                    continue
                stop_prog = float(sp[pos])
                fr = self.route_factor(r, f)
                ro, re_ = f["raw_routes"].get(r, (0.0, 0.0))
                f_route = float(F.route_factor(ro, re_, f["net"]))
                for s in buses.values():
                    age = now - s["ts"]
                    if age > STALE_S or s["prog"] > stop_prog + PAST_STOP_M:
                        continue
                    b = self._bucket(s["ts"])
                    cs = self.cum_time(r, b, "scaled")
                    pr = min(s["prog"], stop_prog)
                    rem_hist = float(np.interp(stop_prog, sp, cs) - np.interp(pr, sp, cs)) * fr
                    nxt = int(np.searchsorted(sp, s["prog"] + F.AHEAD_M, "right"))
                    h = pos - nxt + 1
                    c = {"r": r, "s": s, "age": age, "rem_hist": rem_hist, "h": h, "pos": pos,
                         "stop_prog": stop_prog, "f_route": f_route, "e_route": re_, "bucket": b,
                         "rem": rem_hist, "source": "history"}
                    cand.append(c)
            if use_ai and self.eta_model is not None:
                self._apply_model(cand, f)
        out = []
        for c in cand:
            s = c["s"]
            eta = max(c["rem"] - c["age"], 0.0)
            if eta > max_eta_s:
                continue
            out.append({
                "route": self.net.route_short[c["r"]], "route_id": int(self.net.route_ids[c["r"]]),
                "description": self.net.route_desc[c["r"]],
                "vehicle": s["name"], "vehicle_id": s["name"],
                "eta_s": round(eta), "eta_min": round(eta / 60, 1), "eta_source": c["source"],
                "stops_away": max(0, c["h"]),
                "remaining_m": int(max(c["stop_prog"] - s["prog"], 0)),
                "ping_age_s": int(c["age"]),
                "standing": bool(now - s["move_ts"] > 120),
                "lat": s["lat"], "lon": s["lon"],
            })
        out.sort(key=lambda b: b["eta_s"])
        return out[:max_n]

    def _apply_model(self, cand, f):
        """Replace the history ETA with history ETA + model correction where the model applies."""
        use = [c for c in cand if 1 <= c["h"] <= F.HMAX and c["stop_prog"] - c["s"]["prog"] > 0]
        if not use:
            return
        cols = defaultdict(list)
        o_n, e_n = f["raw_net"]
        for c in use:
            s, r = c["s"], c["r"]
            sp = self.net.stop_prog[r]
            cr = self.cum_time(r, c["bucket"], "raw")
            csp = self.cum_time(r, c["bucket"], "spread")
            sprog = c["stop_prog"]
            base = float(np.interp(sprog, sp, cr) - np.interp(s["prog"], sp, cr))
            spread = float(np.interp(sprog, sp, csp) - np.interp(s["prog"], sp, csp))
            hist = list(s["hist"])
            own_o, own_e = sum(a for a, _ in hist), sum(b for _, b in hist)
            route_len = float(sp[-1])
            cols["base"].append(base)
            cols["hist"].append(c["rem_hist"])
            cols["h"].append(c["h"])
            cols["dist_rem"].append(sprog - s["prog"])
            cols["spread"].append(spread)
            cols["frac_route"].append(sprog / route_len if route_len > 0 else 0.0)
            cols["route_len"].append(route_len)
            cols["route_speed"].append(route_len / cr[-1] if cr[-1] > 0 else np.nan)
            cols["ts"].append(s["ts"])
            cols["f_net"].append(float(F.net_factor(o_n, e_n)))
            cols["e_net"].append(e_n)
            cols["f_route"].append(c["f_route"])
            cols["e_route"].append(c["e_route"])
            cols["own_obs"].append(own_o)
            cols["own_exp"].append(own_e)
            cols["standing_s"].append(s["ts"] - s["move_ts"])
        X = F.build_matrix({k: np.asarray(v, dtype=float) for k, v in cols.items()})
        try:
            resid = self.eta_model.predict(X)
        except Exception as e:  # noqa: BLE001
            self.eta_info = {**self.eta_info, "loaded": False, "reason": f"prediction failed: {e}"}
            self.eta_model = None
            return
        for c, b, d in zip(use, cols["hist"], resid):
            c["rem"] = max(b + float(d), 0.0)
            c["source"] = "model"

    # -- waiting time for the journey search
    def wait_model(self, now, default_waits):
        return WaitModel(self, now, default_waits)

    def status(self, now=None):
        now = now or time.time()
        f = self.factors(now)
        with self.lock:
            n_buses = sum(len(d) for d in self.buses.values())
            unres = dict(self.unresolved)
        by_route = defaultdict(int)
        for rid in unres.values():
            by_route[rid] += 1
        return {
            "buses_live": max(self.live_total, n_buses + len(unres)),
            "unresolved_buses": len(unres), "unresolved_routes": len(by_route),
            "unresolved_top_routes": [{"route_id": r, "buses": n} for r, n in
                                      sorted(by_route.items(), key=lambda x: -x[1])[:5]],
            "buses_tracked": n_buses, "routes_tracked": len(self.buses),
            "warm": f["warm"], "network_factor": round(f["net"], 3), "events_10min": f["n_events"],
            "expected_bus_seconds_10min": f["expected_s"], "updates": self.updates,
            "last_update_age_s": round(now - self.last_update, 1) if self.last_update else None,
            "skipped": dict(self.counts), "eta_model": self.eta_info,
        }


class WaitModel:
    """
    Waiting time at a boarding stop from the live buses on that route (history ETA x live ratio).
    wait_info(route, position, arrive_s) -> (seconds, vehicle name or None, 'live' | 'timetable' | None)
    arrive_s is when the rider reaches the stop, in seconds from now.
    A live wait is capped at 3 x the timetable wait, because buses that have not left the
    terminal yet are not in the feed.
    """

    def __init__(self, live, now, default_waits):
        self.default = default_waits
        self.routes = {}
        f = live.factors(now)
        with live.lock:
            for r, buses in live.buses.items():
                g = live.geom(r)
                sp = g.stop_prog
                fr = live.route_factor(r, f)
                prog, age, names = [], [], []
                for s in buses.values():
                    a = now - s["ts"]
                    if a > STALE_S:
                        continue
                    prog.append(s["prog"]); age.append(a); names.append(s["name"])
                if not prog:
                    continue
                cs = live.cum_time(r, live._bucket(now), "scaled")
                prog = np.asarray(prog)
                t0 = np.interp(np.minimum(prog, sp[-1]), sp, cs)
                self.routes[r] = {"prog": prog, "age": np.asarray(age), "names": names,
                                  "t0": t0, "cs": cs, "sp": sp, "fr": fr}

    def has_route(self, r):
        return r in self.routes

    def wait_info(self, r, pos, arrive_s):
        base = self.default[r]
        d = self.routes.get(r)
        if d is not None:
            sp = d["sp"]
            ok = d["prog"] <= sp[pos] + PAST_STOP_M
            eta = (d["cs"][pos] - d["t0"]) * d["fr"] - d["age"]
            good = ok & (eta >= arrive_s + BOARD_MARGIN_S)
            if good.any():
                i = int(np.argmin(np.where(good, eta, np.inf)))
                w = float(eta[i] - arrive_s)
                if base is None or w <= 3 * base:
                    return w, d["names"][i], "live"
        if base is not None:
            return base, None, "timetable"
        return None, None, None
