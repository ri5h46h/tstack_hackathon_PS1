"""Tests for road shapes, vehicle names, live waiting time and the ETA model hook."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_planner as T  # noqa: E402
import test_live as TL  # noqa: E402
from planner import features as F  # noqa: E402


def test_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / "gtfs"
        T.write_gtfs(g)
        trips = pd.read_csv(g / "trips.txt")
        trips["shape_id"] = np.where(trips["route_id"] == 10, "S10", "")
        trips.to_csv(g / "trips.txt", index=False)
        lon = lambda m: T.LON0 + m / TL.M_PER_DEG_LON  # noqa: E731
        # the road bends 0.002 degrees north between stop 2 (1000 m) and stop 3 (2000 m)
        pts = [(T.LAT, lon(0)), (T.LAT, lon(1000)), (T.LAT + 0.002, lon(1500)), (T.LAT, lon(2000)), (T.LAT, lon(2100))]
        pd.DataFrame([("S10", a, b, i, 0) for i, (a, b) in enumerate(pts)],
                     columns=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence",
                              "shape_dist_traveled"]).to_csv(g / "shapes.txt", index=False)
        T.write_eta(Path(tmp) / "data")
        from planner.search import Planner
        p = Planner(str(g), str(Path(tmp) / "data"), rebuild=True)
        r = p.net.route_index(10)
        assert p.net.has_shape[r]
        path = p.net.route_path(r, 1, 2)
        assert max(a for a, _ in path) > T.LAT + 0.0015, "path should follow the bend"
        assert p.net.seg_dist[r][1] > 1000, "segment length should follow the road, not a straight line"
        print("shapes: path points", len(path), " segment m", round(float(p.net.seg_dist[r][1])))


def test_wait_names_and_model():
    with tempfile.TemporaryDirectory() as tmp:
        p, m = TL.build(Path(tmp))
        NOW = TL.NOW
        one = [{"vehicle_id": "DL1PX0001", "name": "DL1PX0001", "route_id": 10, "trip_id": "t", "lat": T.LAT,
                "lon": TL.lon_at(300), "ts": int(NOW - 10), "speed_rep": None}]
        m.update(one, NOW)
        r = p.net.route_index(10)
        waits = p.net.waits(9)
        wm = m.wait_model(NOW, waits)
        w, veh, src = wm.wait_info(r, 1, 0.0)               # bus is 700 m before stop 2
        print("live wait", round(w), veh, src)
        assert src == "live" and veh == "DL1PX0001" and 50 < w < 130, (w, veh, src)
        w2, veh2, src2 = wm.wait_info(r, 1, 600.0)           # rider arrives after the bus has gone
        assert src2 == "timetable" and veh2 is None, (w2, veh2, src2)
        # a route with no live bus keeps the timetable wait
        assert wm.wait_info(p.net.route_index(20), 0, 0.0)[2] == "timetable"

        b = m.approaching(p.net.stop_index(2), None, now=NOW)[0]
        assert b["vehicle"] == "DL1PX0001" and b["eta_source"] == "history"
        base_eta = b["eta_s"]

        class Fake:                                         # stands in for the trained model
            def predict(self, X):
                assert X.shape[1] == len(F.FEATURES)
                return np.full(len(X), -30.0)
        m.eta_model = Fake()
        b2 = m.approaching(p.net.stop_index(2), None, now=NOW)[0]
        print("history eta", base_eta, "-> model eta", b2["eta_s"])
        assert b2["eta_source"] == "model" and abs((base_eta - b2["eta_s"]) - 30) <= 1


def test_features_parity():
    ok = F.interval_ok([30, 5, 30], [500, 500, 100], [700, 520, 700], [5000, 5000, 5000])
    assert ok.tolist() == [True, False, False]
    assert float(F.net_factor(0.0, 10.0)) == 1.0            # not enough data: neutral
    assert abs(float(F.net_factor(4000.0, 3000.0)) - 1.333) < 0.01


if __name__ == "__main__":
    for fn in (test_shapes, test_wait_names_and_model, test_features_parity):
        print("---", fn.__name__)
        fn()
    print("PASS")
