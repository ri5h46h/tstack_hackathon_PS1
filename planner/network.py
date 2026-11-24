"""
network.py  -  the bus network built from the static GTFS folder.

Holds: stops, one pattern (ordered stop list) per route, the road shape of each route
(from shapes.txt, straight lines between stops when a shape is missing), how far along
the shape each stop is, walking links between nearby stops, and how many trips each
route starts per hour (for waiting times).

Route geometry follows the same rules as build/build_arrivals.py, so a bus's progress
here means the same thing as in the pings and arrivals files:
  - metres, flat-earth projection
  - stops are placed along the shape in order: each stop goes on the nearest segment
    whose end is no more than 50 m behind the previous stop, and progress never decreases

Building takes a few seconds to a minute on the full Delhi GTFS, so the result is
cached in <data>/.network_cache.pkl and rebuilt only when a GTFS file changes.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

WALK_SPEED = 1.25      # m/s, about 4.5 km/h
WALK_DETOUR = 1.3      # straight line -> real walking distance
BUS_DETOUR = 1.25      # straight line between stops -> road distance (only for routes without a shape)
FOOT_RADIUS_M = 400    # walking links between stops closer than this
LAT0 = 28.6            # Delhi, for the flat-earth distance formula
BACK_TOL_M = 50        # a stop may sit this far behind the previous stop on the shape
SHAPE_MAX_STOP_DIST_M = 250   # stops farther than this from their shape make the shape untrusted
CACHE_VERSION = 2


def to_xy(lat, lon):
    """Flat-earth metres around Delhi. Good to well under 1% inside the city."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - 77.2) * 111320.0 * np.cos(np.radians(LAT0))
    y = (lat - LAT0) * 110540.0
    return x, y


def _hms_to_s(series):
    p = series.str.split(":", expand=True).astype(int)
    return p[0] * 3600 + p[1] * 60 + p[2]


def _segments(x, y):
    ax, ay = x[:-1], y[:-1]
    vx, vy = x[1:] - ax, y[1:] - ay
    l2 = np.maximum(vx ** 2 + vy ** 2, 1e-6)
    return ax, ay, vx, vy, l2, np.sqrt(l2)


def place_stops_on_line(x, y, sx, sy):
    """
    Progress (metres along the line) of each stop, and its distance from the line.
    Same rule as build_arrivals: nearest segment whose end is at most BACK_TOL_M behind the
    previous stop's progress, then never decreasing.
    """
    ax, ay, vx, vy, l2, seglen = _segments(x, y)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    prev = 0.0
    prog, dist = [], []
    for px, py in zip(sx, sy):
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / l2, 0.0, 1.0)
        d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
        ok = cum[1:] >= prev - BACK_TOL_M
        if ok.any():
            d = np.where(ok, d, np.inf)
        i = int(np.argmin(d))
        p = max(float(cum[i] + t[i] * seglen[i]), prev)
        prog.append(p)
        dist.append(float(d[i]))
        prev = p
    return np.array(prog), np.array(dist), cum


class Network:
    def __init__(self, gtfs_dir="gtfs", cache_dir="data", rebuild=False):
        self.gtfs = Path(gtfs_dir)
        cache = Path(cache_dir) / ".network_cache.pkl"
        sig = self._signature()
        if not rebuild and cache.exists():
            try:
                d = pickle.load(open(cache, "rb"))
                if d.get("sig") == sig and d.get("version") == CACHE_VERSION:
                    self.__dict__.update(d["data"])
                    return
            except Exception:
                pass
        self._build()
        try:
            Path(cache_dir).mkdir(exist_ok=True)
            pickle.dump({"sig": sig, "version": CACHE_VERSION, "data": self._data()},
                        open(cache, "wb"), protocol=4)
        except Exception as e:  # cache is optional
            print("could not write cache:", e)

    # ------------------------------------------------------------ cache
    def _signature(self):
        sig = []
        for n in ("stops", "routes", "trips", "stop_times", "shapes"):
            f = self.gtfs / f"{n}.txt"
            if f.exists():
                st = f.stat()
                sig.append((n, st.st_size, int(st.st_mtime)))
        return sig

    def _data(self):
        keys = ["stop_ids", "stop_names", "stop_lat", "stop_lon", "stop_x", "stop_y",
                "route_ids", "route_short", "route_desc", "patterns", "seg_dist",
                "stop_routes", "foot", "hourly",
                "geom_lat", "geom_lon", "geom_x", "geom_y", "geom_cum", "stop_prog", "has_shape"]
        return {k: getattr(self, k) for k in keys}

    # ------------------------------------------------------------ build
    def _load_shapes(self, needed):
        f = self.gtfs / "shapes.txt"
        if not f.exists() or not needed:
            return {}
        parts = []
        for chunk in pd.read_csv(f, usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
                                 chunksize=1_000_000):
            parts.append(chunk[chunk["shape_id"].isin(needed)])
        if not parts:
            return {}
        df = pd.concat(parts).sort_values(["shape_id", "shape_pt_sequence"], kind="stable")
        return {sid: (g["shape_pt_lat"].to_numpy(float), g["shape_pt_lon"].to_numpy(float))
                for sid, g in df.groupby("shape_id", sort=False)}

    def _build(self):
        g = self.gtfs
        stops = pd.read_csv(g / "stops.txt", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"])
        routes = pd.read_csv(g / "routes.txt")
        tcols = pd.read_csv(g / "trips.txt", nrows=0).columns
        use = ["route_id", "trip_id"] + (["shape_id"] if "shape_id" in tcols else [])
        trips = pd.read_csv(g / "trips.txt", usecols=use)
        st = pd.read_csv(g / "stop_times.txt",
                         usecols=["trip_id", "departure_time", "stop_id", "stop_sequence"])

        self.stop_ids = stops["stop_id"].to_numpy()
        self.stop_names = stops["stop_name"].astype(str).tolist()
        self.stop_lat = stops["stop_lat"].to_numpy(float)
        self.stop_lon = stops["stop_lon"].to_numpy(float)
        self.stop_x, self.stop_y = to_xy(self.stop_lat, self.stop_lon)
        sid2i = {int(s): i for i, s in enumerate(self.stop_ids)}

        # one representative trip per route: the one with the most stops
        n_stops = st.groupby("trip_id").size()
        trips["n"] = trips["trip_id"].map(n_stops).fillna(0)
        trips = trips[trips["n"] > 0]
        rep = trips.sort_values("n", kind="stable").groupby("route_id").tail(1)
        rep_trip = dict(zip(rep["route_id"], rep["trip_id"]))
        rep_shape = dict(zip(rep["route_id"], rep["shape_id"])) if "shape_id" in rep else {}

        rinfo = routes.set_index("route_id")
        self.route_ids = sorted(rep_trip)
        rid2i = {r: i for i, r in enumerate(self.route_ids)}
        self.route_short = [str(rinfo["route_short_name"].get(r, r)) for r in self.route_ids]
        self.route_desc = [str(rinfo["route_desc"].get(r, "")) for r in self.route_ids]

        st_rep = st[st["trip_id"].isin(set(rep_trip.values()))].sort_values(["trip_id", "stop_sequence"])
        by_trip = {t: grp["stop_id"].to_numpy() for t, grp in st_rep.groupby("trip_id", sort=False)}
        shapes = self._load_shapes({s for s in rep_shape.values() if isinstance(s, str) or pd.notna(s)})

        self.patterns, self.seg_dist = [], []
        self.geom_lat, self.geom_lon, self.geom_x, self.geom_y = [], [], [], []
        self.geom_cum, self.stop_prog, self.has_shape = [], [], []
        for r in self.route_ids:
            ids = by_trip[rep_trip[r]]
            idx = np.array([sid2i[int(s)] for s in ids if int(s) in sid2i], dtype=np.int32)
            self.patterns.append(idx)
            sx, sy = self.stop_x[idx], self.stop_y[idx]

            used = False
            shp = shapes.get(rep_shape.get(r)) if rep_shape else None
            if shp is not None and len(shp[0]) >= 2 and len(idx) >= 2:
                lat, lon = shp
                x, y = to_xy(lat, lon)
                prog, dist, cum = place_stops_on_line(x, y, sx, sy)
                if (dist > SHAPE_MAX_STOP_DIST_M).mean() <= 0.3:
                    used = True
            if used:
                self.seg_dist.append(np.maximum(np.diff(prog), 1.0))
            else:
                lat, lon = self.stop_lat[idx], self.stop_lon[idx]
                x, y = sx, sy
                cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) if len(idx) else np.zeros(1)
                prog = cum.copy()
                self.seg_dist.append(np.maximum(np.diff(prog) * BUS_DETOUR, 1.0))
            self.geom_lat.append(np.asarray(lat, float)); self.geom_lon.append(np.asarray(lon, float))
            self.geom_x.append(np.asarray(x, float)); self.geom_y.append(np.asarray(y, float))
            self.geom_cum.append(np.asarray(cum, float)); self.stop_prog.append(np.asarray(prog, float))
            self.has_shape.append(used)

        self.stop_routes = [[] for _ in self.stop_ids]
        for r, pat in enumerate(self.patterns):
            seen = set()
            for pos, s in enumerate(pat):
                if int(s) not in seen:  # a stop visited twice is boarded at its first visit
                    seen.add(int(s))
                    self.stop_routes[int(s)].append((r, pos))

        # trips starting per hour of day, per route
        first = st.loc[st.groupby("trip_id")["stop_sequence"].idxmin(), ["trip_id", "departure_time"]]
        first = first.merge(trips[["trip_id", "route_id"]], on="trip_id")
        first["h"] = (_hms_to_s(first["departure_time"]) // 3600) % 24
        self.hourly = np.zeros((len(self.route_ids), 24))
        for (r, h), c in first.groupby(["route_id", "h"]).size().items():
            if r in rid2i:
                self.hourly[rid2i[r], int(h)] = c

        # walking links between nearby stops
        self.foot = [[] for _ in self.stop_ids]
        n = len(self.stop_ids)
        for a in range(0, n, 400):
            b = min(n, a + 400)
            dx = self.stop_x[a:b, None] - self.stop_x[None, :]
            dy = self.stop_y[a:b, None] - self.stop_y[None, :]
            d = np.hypot(dx, dy)
            ii, jj = np.nonzero(d < FOOT_RADIUS_M)
            for i, j in zip(ii, jj):
                if a + i != j:
                    self.foot[a + i].append((int(j), float(d[i, j] * WALK_DETOUR / WALK_SPEED)))

    # ------------------------------------------------------------ queries
    def dist_to(self, lat, lon):
        """Straight-line metres from a point to every stop."""
        x, y = to_xy(lat, lon)
        return np.hypot(self.stop_x - x, self.stop_y - y)

    def xy(self, lat, lon):
        x, y = to_xy(lat, lon)
        return float(x), float(y)

    def stop_index(self, stop_id):
        """Position of a GTFS stop_id in the stop arrays (None if unknown)."""
        if not hasattr(self, "_sid2i"):
            self._sid2i = {int(s): i for i, s in enumerate(self.stop_ids)}
        try:
            return self._sid2i.get(int(stop_id))
        except (TypeError, ValueError):
            return None

    def route_index(self, route_id):
        """Position of a GTFS route_id among the routes that have a pattern (None if unknown)."""
        if not hasattr(self, "_rid2i"):
            self._rid2i = {int(r): i for i, r in enumerate(self.route_ids)}
        try:
            return self._rid2i.get(int(route_id))
        except (TypeError, ValueError):
            return None

    def route_path(self, r, bp, ap):
        """[[lat, lon], ...] along the road from stop position bp to stop position ap of route r."""
        sp = self.stop_prog[r]
        cum, lat, lon = self.geom_cum[r], self.geom_lat[r], self.geom_lon[r]
        p0, p1 = float(sp[bp]), float(sp[ap])
        if len(cum) < 2 or p1 <= p0:
            pat = self.patterns[r]
            return [[float(self.stop_lat[i]), float(self.stop_lon[i])] for i in pat[bp:ap + 1]]
        lo = int(np.searchsorted(cum, p0, "right"))
        hi = int(np.searchsorted(cum, p1, "left"))
        pts = [[float(np.interp(p0, cum, lat)), float(np.interp(p0, cum, lon))]]
        pts += [[float(lat[i]), float(lon[i])] for i in range(lo, hi)]
        pts.append([float(np.interp(p1, cum, lat)), float(np.interp(p1, cum, lon))])
        return [[round(a, 5), round(b, 5)] for a, b in pts]

    def walk_seconds(self, a, b):
        return float(np.hypot(self.stop_x[a] - self.stop_x[b], self.stop_y[a] - self.stop_y[b])
                     * WALK_DETOUR / WALK_SPEED)

    def waits(self, hour):
        """Expected waiting time per route (seconds) for a trip starting this hour.
        None where the route has (almost) no trips around that hour."""
        h = self.hourly
        c = (h[:, (hour - 1) % 24] + 2 * h[:, hour % 24] + h[:, (hour + 1) % 24]) / 4.0
        out = []
        for x in c:
            out.append(None if x < 0.5 else min(1500.0, 1800.0 / x))
        return out

    def summary(self):
        n_edges = sum(max(0, len(p) - 1) for p in self.patterns)
        shaped = int(sum(self.has_shape))
        return (f"{len(self.stop_ids):,} stops, {len(self.route_ids):,} routes "
                f"({shaped:,} with road shapes), {n_edges:,} stop-to-stop edges")
