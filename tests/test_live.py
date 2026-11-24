"""
Tests for planner/live.py with simulated buses (no network, no API keys needed).
Run from the repo root:  python tests/test_live.py

Uses the same tiny network as test_planner.py: route 10 has stops 1-2-3, 1 km apart,
and the ETA table says each 1 km hop takes 150 s in the morning (6.7 m/s).
"""
import datetime as dt
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_planner as T  # noqa: E402
from planner import live as L  # noqa: E402
from planner.search import Planner  # noqa: E402

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime(2026, 11, 24, 9, 30, tzinfo=IST).timestamp()      # morning bucket
M_PER_DEG_LON = 111320 * 0.8786                                         # at 28.6 N, about 97,800


def lon_at(meters_from_stop1):
    return T.LON0 + meters_from_stop1 / M_PER_DEG_LON


def fleet(speed_mps, n_buses=40, n_pings=7, gap=30, start_m=300, route_id=10):
    """Rows for n buses driving east on route 10 at a constant speed, last ping at NOW - 10 s."""
    rows = []
    for b in range(n_buses):
        for k in range(n_pings):
            ts = NOW - 10 - (n_pings - 1 - k) * gap - (b % 5)
            m = start_m + speed_mps * k * gap
            rows.append({"vehicle_id": f"DL1PD{b:04d}", "route_id": route_id, "trip_id": "t",
                         "lat": T.LAT, "lon": lon_at(m), "ts": int(ts), "speed_rep": None})
    return sorted(rows, key=lambda r: r["ts"])


def build(tmp):
    T.write_gtfs(tmp / "gtfs")
    T.write_eta(tmp / "data")
    p = Planner(str(tmp / "gtfs"), str(tmp / "data"))
    m = L.LiveModel(p)
    p.attach_live(m)
    return p, m


def test_factor_and_approaching():
    with tempfile.TemporaryDirectory() as tmp:
        p, m = build(Path(tmp))
        normal = 1000 / 150.0                   # what the table says
        m.update(fleet(normal), NOW)
        f = m.factors(NOW)
        print("normal traffic:  warm", f["warm"], " factor", round(f["net"], 3))
        assert f["warm"] and 0.9 < f["net"] < 1.1, f

        # slower traffic: same table, buses move at 75 % of the usual speed
        p2, m2 = build(Path(tmp) / "b")
        m2.update(fleet(normal * 0.75), NOW)
        f2 = m2.factors(NOW)
        print("slow traffic:    factor", round(f2["net"], 3), " (expect about 1.33)")
        assert 1.2 < f2["net"] < 1.5, f2

        # a bus 400 m past stop 1 heading to stop 3 (2000 m): table time = 240 s
        one = [{"vehicle_id": "DL1PX0001", "route_id": 10, "trip_id": "t", "lat": T.LAT,
                "lon": lon_at(400), "ts": int(NOW - 20), "speed_rep": None}]
        p3, m3 = build(Path(tmp) / "c")
        m3.update(one, NOW)
        stop3 = p3.net.stop_index(3)
        buses = m3.approaching(stop3, None, now=NOW)
        print("approaching stop 3:", [(b["route"], b["eta_s"], b["stops_away"]) for b in buses])
        assert len(buses) == 1 and buses[0]["route"] == "R10"
        assert 200 <= buses[0]["eta_s"] <= 240, buses[0]      # 240 s of table time minus 20 s since the ping
        assert buses[0]["stops_away"] == 2

        # a bus already past the stop must not be listed
        past = [{"vehicle_id": "DL1PX0002", "route_id": 10, "trip_id": "t", "lat": T.LAT,
                 "lon": lon_at(1600), "ts": int(NOW - 5), "speed_rep": None}]
        m3.update(past, NOW)
        got = m3.approaching(p3.net.stop_index(2), None, now=NOW)
        assert all(b["vehicle"] != "0002" for b in got), "bus past the stop was listed"

        # off-route bus is ignored
        off = [{"vehicle_id": "DL1PX0003", "route_id": 10, "trip_id": "t", "lat": T.LAT + 0.05,
                "lon": lon_at(500), "ts": int(NOW - 5), "speed_rep": None}]
        m3.update(off, NOW)
        assert m3.counts["off_route"] >= 1


def test_plan_uses_live():
    with tempfile.TemporaryDirectory() as tmp:
        p, m = build(Path(tmp))
        normal = 1000 / 150.0
        origin = (T.LAT, T.LON0 - 100 / M_PER_DEG_LON)
        dest = (T.LAT, T.LON0 + 4 * 1000 / M_PER_DEG_LON + 150 / M_PER_DEG_LON + 100 / M_PER_DEG_LON)
        # use a morning departure in the past via depart="09:30": live is not applied
        r0 = p.plan(origin, dest, "09:30")
        assert r0["live"]["applied"] is False
        base = r0["options"][0]["total_min"]

        # for "now" trips the planner reads the real clock, so check the factor path directly
        m.update(fleet(normal * 0.75), NOW)
        f = m.factors(NOW)
        slow = p.plan(origin, dest, "09:30", speed_factor=f["net"])
        print(f"base {base} min -> with traffic factor {f['net']:.2f}: {slow['options'][0]['total_min']} min")
        assert slow["options"][0]["total_min"] > base


def test_decode_with_fake_bindings():
    class Msg:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self._set = set(kw)

        def HasField(self, name):
            return name in self._set

    now = 1_800_000_000
    good = Msg(vehicle=Msg(
        trip=Msg(route_id="1942", trip_id="1942_x"), vehicle=Msg(id="DL1PD1234", label=""),
        position=Msg(latitude=28.7, longitude=77.1, speed=5.0), timestamp=now - 10), id="e1")
    bad_box = Msg(vehicle=Msg(trip=Msg(route_id="1", trip_id=""), vehicle=Msg(id="X", label=""),
                              position=Msg(latitude=0.0, longitude=0.0), timestamp=now - 10), id="e2")
    stale = Msg(vehicle=Msg(trip=Msg(route_id="1", trip_id=""), vehicle=Msg(id="Y", label=""),
                            position=Msg(latitude=28.6, longitude=77.2), timestamp=now - 5000), id="e3")
    no_vehicle = Msg(id="e4")

    class Feed:
        def __init__(self):
            self.entity = [good, bad_box, stale, no_vehicle]
            self.header = Msg(timestamp=now)

        def ParseFromString(self, content):
            pass

    mod = types.ModuleType("google.transit.gtfs_realtime_pb2")
    mod.FeedMessage = Feed
    pkg = types.ModuleType("google.transit")
    pkg.gtfs_realtime_pb2 = mod
    sys.modules["google.transit"] = pkg
    sys.modules["google.transit.gtfs_realtime_pb2"] = mod
    try:
        rows = L.decode_vehicle_positions(b"x", "tstack", now=now)
        assert len(rows) == 1, rows
        assert rows[0]["route_id"] == 1942 and rows[0]["vehicle_id"] == "DL1PD1234"
        assert rows[0]["speed_rep"] == 5.0
        rows2 = L.decode_vehicle_positions(b"x", "otd", now=now)
        assert rows2[0]["speed_rep"] is None, "OTD speed must be ignored"
    finally:
        sys.modules.pop("google.transit", None)
        sys.modules.pop("google.transit.gtfs_realtime_pb2", None)


def test_feed_fallback():
    feed = L.LiveFeed("tskey", "otdkey", primary="tstack")
    calls = []
    big = [{"vehicle_id": str(i), "route_id": 1, "trip_id": "", "lat": 28.6, "lon": 77.2, "ts": 1, "speed_rep": None}
           for i in range(300)]

    def fake_get(src):
        calls.append(src)
        if src == "tstack":
            raise RuntimeError("tstack: down")
        return b"ok"

    feed._get = fake_get
    orig = L.decode_vehicle_positions
    L.decode_vehicle_positions = lambda content, source, now=None: big
    try:
        assert feed.refresh_once() is False and feed.active == "tstack"     # first failure
        assert feed.refresh_once() is False and feed.active == "otd"        # second: switch
        assert feed.refresh_once() is True and feed.source == "otd"
        assert feed.status()["buses_in_feed"] == 300

        # a snapshot with fewer than 20 % of the previous buses is rejected
        L.decode_vehicle_positions = lambda content, source, now=None: big[:10]
        feed.cycle = 5                       # avoid the "retry the main feed" poll (every 4th)
        assert feed.refresh_once() is False and feed.rejected == 1
        assert len(feed.rows) == 300, "old snapshot must be kept"
    finally:
        L.decode_vehicle_positions = orig
    no_keys = L.LiveFeed(None, None)
    assert not no_keys.available


def test_env_loader():
    import os
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / ".env"
        p.write_text('# comment\nTSTACK_API_KEY="abc123"\nexport OTD_API_KEY=xyz\n')
        os.environ.pop("TSTACK_API_KEY", None)
        os.environ.pop("OTD_API_KEY", None)
        L.load_env(p)
        assert os.environ["TSTACK_API_KEY"] == "abc123" and os.environ["OTD_API_KEY"] == "xyz"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name}")
            fn()
    print("PASS")
