"""
Tiny end-to-end test of the planner on a made-up network.
Run from the repo root:  python tests/test_planner.py

Network (stops on a west-east line, 1 km apart, around Delhi):
    route 10:  stop1 - stop2 - stop3            (runs every 10 min)
    route 20:  stop4 - stop5 - stop6            (runs every 15 min)
    stop3 and stop4 are 150 m apart, so a rider can walk between them.
A trip from near stop1 to near stop6 therefore needs route 10, a walk, then route 20.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planner.search import Planner  # noqa: E402

LAT = 28.6
LON0 = 77.2
DLON = 1000 / (111320 * np.cos(np.radians(LAT)))  # about 1 km in longitude


def write_gtfs(d):
    d.mkdir(parents=True)
    lons = [LON0 + i * DLON for i in range(3)] + [LON0 + 2 * DLON + 150 / 97800 + j * DLON for j in range(3)]
    stops = pd.DataFrame({"stop_id": [1, 2, 3, 4, 5, 6], "stop_code": "x",
                          "stop_name": [f"Stop {i}" for i in range(1, 7)],
                          "stop_lat": LAT, "stop_lon": lons})
    stops.to_csv(d / "stops.txt", index=False)
    pd.DataFrame({"agency_id": "DTC", "route_id": [10, 20], "route_long_name": ["A", "B"],
                  "route_short_name": ["R10", "R20"], "route_desc": ["west line", "east line"],
                  "route_type": 3}).to_csv(d / "routes.txt", index=False)
    trips, rows = [], []
    for route, stop_ids, every in ((10, [1, 2, 3], 10), (20, [4, 5, 6], 15)):
        for h in range(5, 23):
            for m in range(0, 60, every):
                tid = f"{route}_{h:02d}_{m:02d}"
                trips.append((route, 1, tid))
                for seq, s in enumerate(stop_ids):
                    mm = m + seq * 3
                    t = f"{h + mm // 60:02d}:{mm % 60:02d}:00"
                    rows.append((tid, t, t, s, seq))
    pd.DataFrame(trips, columns=["route_id", "service_id", "trip_id"]).to_csv(d / "trips.txt", index=False)
    pd.DataFrame(rows, columns=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
                 ).to_csv(d / "stop_times.txt", index=False)


def write_eta(d):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for route, pairs in ((10, [(1, 2), (2, 3)]), (20, [(4, 5), (5, 6)])):
        for a, b in pairs:
            for bucket, sec in (("morning", 150.0), ("midday", 180.0), ("all", 165.0)):
                rows.append((route, a, b, bucket, 30, sec, sec - 20, sec + 20, 1000.0))
    pd.DataFrame(rows, columns=["route_id", "from_stop", "to_stop", "bucket", "n", "median_s",
                                "p25_s", "p75_s", "dist_m"]).to_csv(d / "eta_bucketed.csv.gz", index=False)
    info = {"buckets": [["morning", 6, 11], ["midday", 11, 16], ["pm_peak", 16, 20], ["night", 20, 6]],
            "split_daytype": False, "blend_k": 20, "scale_bucket": {}, "scale_all": 1.0,
            "v_route_bucket": {}, "v_route": {}, "v_global_bucket": {}, "v_global": 6.0}
    json.dump(info, open(d / "eta_bucketed_info.json", "w"))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_gtfs(tmp / "gtfs")
        write_eta(tmp / "data")
        p = Planner(str(tmp / "gtfs"), str(tmp / "data"))
        print("network:", p.net.summary())

        # origin 100 m west of stop 1, destination 100 m east of stop 6
        origin = (LAT, LON0 - 100 / 97800)
        dest = (LAT, LON0 + 2 * DLON + 150 / 97800 + 2 * DLON + 100 / 97800)
        res = p.plan(origin, dest, "09:30", max_transfers=2)
        print("bucket:", res["bucket"], " options:", len(res["options"]))
        assert res["options"], "no journey found"
        best = res["options"][0]
        buses = [leg for leg in best["legs"] if leg["type"] == "bus"]
        for leg in best["legs"]:
            print("  ", leg)
        print("total", best["total_min"], "min,", best["transfers"], "transfer(s), confidence", best["confidence"])
        assert [b["route"] for b in buses] == ["R10", "R20"], "expected R10 then R20"
        assert best["transfers"] == 1
        # two rides of about 2 x 150 s, two waits of about 5 and 7.5 min, plus walking
        assert 15 < best["total_min"] < 40, best["total_min"]
        assert best["confidence"] == "high"

        # evening has no data in the table: fallback must still give an answer
        res2 = p.plan(origin, dest, "18:00")
        assert res2["options"], "no journey in evening"
        print("evening total", res2["options"][0]["total_min"], "min, confidence",
              res2["options"][0]["confidence"])

        # a slower network gives a longer trip
        res3 = p.plan(origin, dest, "09:30", speed_factor=1.5)
        assert res3["options"][0]["total_min"] > best["total_min"]

        # no service at 03:00
        res4 = p.plan(origin, dest, "03:00")
        assert not res4["options"], "should be no service at 3 AM"
    print("PASS")


if __name__ == "__main__":
    main()
