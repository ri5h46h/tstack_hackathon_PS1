"""
app.py  -  the journey planner API and the demo web page.

Run from the repo root:
    uv run uvicorn api.app:app --reload
Then open  http://127.0.0.1:8000/       (the web page)
and        http://127.0.0.1:8000/docs   (the API, to try it by hand)

Live data: put TSTACK_API_KEY and OTD_API_KEY in .env. Without keys the planner
still works from the learned data; the live parts just stay off.
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from planner.live import LiveFeed, LiveModel, load_env  # noqa: E402
from planner.search import Planner  # noqa: E402

state = {"planner": None, "feed": None, "model": None, "live_error": None}


@asynccontextmanager
async def lifespan(app):
    load_env(ROOT / ".env")
    state["planner"] = Planner(str(ROOT / "gtfs"), str(ROOT / "data"))
    try:
        feed = LiveFeed.from_env()
        if feed.available:
            model = LiveModel(state["planner"], ROOT / "models" / "eta_model.joblib")
            feed.add_callback(model.update)
            state["planner"].attach_live(model)
            feed.start()
            state["feed"], state["model"] = feed, model
        else:
            state["live_error"] = "no API keys found (TSTACK_API_KEY / OTD_API_KEY)"
    except Exception as e:  # noqa: BLE001  (live is optional, never stop the planner)
        state["live_error"] = f"live feed could not start: {e}"
    yield
    if state["feed"]:
        state["feed"].stop()


DESCRIPTION = """
Journey planner for Delhi buses. Ride times come from our own collected bus-location data
(a learned stop-to-stop travel-time table by time of day); for trips starting **now** they are
corrected with live GTFS-realtime bus positions, and the waiting time is the time until the next
live bus reaches the boarding stop.

Main endpoint: `GET /journeys?from=28.6139,77.2090&to=stop:1234&depart=now`
"""

app = FastAPI(title="Delhi bus journey planner", version="1.0", description=DESCRIPTION, lifespan=lifespan)


def parse_place(text, name):
    """'lat,lon' or 'stop:<stop_id>' -> (lat, lon)."""
    t = text.strip()
    try:
        if t.lower().startswith("stop:"):
            i = planner().net.stop_index(int(t[5:]))
            if i is None:
                raise HTTPException(404, f"{name}: unknown stop_id")
            net = planner().net
            return float(net.stop_lat[i]), float(net.stop_lon[i])
        lat, lon = (float(x) for x in t.split(","))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, f"{name} must be 'lat,lon' (e.g. 28.61,77.21) or 'stop:<stop_id>'")
    if not (28.0 <= lat <= 29.2 and 76.5 <= lon <= 77.8):
        raise HTTPException(400, f"{name} is outside the Delhi area covered by the data")
    return lat, lon


def planner():
    return state["planner"]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    p = planner()
    return {"status": "ok", "network": p.net.summary(), "eta_days": p.lk.info.get("days", []),
            "loaded_in_s": round(p.load_s, 1)}


@app.get("/live/status")
def live_status():
    if state["feed"] is None:
        return {"enabled": False, "reason": state["live_error"] or "not started"}
    return {"enabled": True, "feed": state["feed"].status(), "model": state["model"].status()}


@app.get("/live/stop")
def live_stop(stop_id: int, max_buses: int = Query(8, ge=1, le=20)):
    """Buses approaching one stop, from all routes that serve it."""
    net = planner().net
    i = net.stop_index(stop_id)
    if i is None:
        raise HTTPException(404, "unknown stop_id")
    buses = state["model"].approaching(i, None, max_n=max_buses) if state["model"] else []
    return {"stop": {"stop_id": stop_id, "name": net.stop_names[i],
                     "lat": float(net.stop_lat[i]), "lon": float(net.stop_lon[i]),
                     "routes": len(net.stop_routes[i])},
            "live": state["model"] is not None, "buses": buses, "server_time": time.time()}


@app.get("/plan")
def plan(from_lat: float = Query(..., ge=28.0, le=29.2), from_lon: float = Query(..., ge=76.5, le=77.8),
         to_lat: float = Query(..., ge=28.0, le=29.2), to_lon: float = Query(..., ge=76.5, le=77.8),
         depart: str = Query("now", description="HH:MM in IST, or 'now'"),
         max_transfers: int = Query(2, ge=0, le=3),
         live: bool = Query(True, description="use live traffic and show approaching buses (only for 'now')"),
         speed_factor: float = Query(1.0, gt=0.3, lt=3.0,
                                     description="1.0 = normal. Above 1 means slower than usual.")):
    if depart != "now":
        try:
            hh, mm = depart.split(":")[:2]
            assert 0 <= int(hh) < 24 and 0 <= int(mm) < 60
        except Exception:
            raise HTTPException(400, "depart must be HH:MM (24-hour, IST) or 'now'")
    return planner().plan((from_lat, from_lon), (to_lat, to_lon), depart,
                          max_transfers=max_transfers, speed_factor=speed_factor, use_live=live)


def _journeys(from_: str, to: str, depart: str, max_transfers: int, live: bool, max_options: int):
    a, b = parse_place(from_, "from"), parse_place(to, "to")
    if depart != "now":
        try:
            hh, mm = depart.split(":")[:2]
            assert 0 <= int(hh) < 24 and 0 <= int(mm) < 60
        except Exception:
            raise HTTPException(400, "depart must be HH:MM (24-hour, IST) or 'now'")
    res = planner().plan(a, b, depart, max_transfers=max_transfers, use_live=live, max_options=max_options)
    return {"from": {"lat": a[0], "lon": a[1]}, "to": {"lat": b[0], "lon": b[1]}, **res}


@app.get("/journeys", summary="Journey options between two places",
         description="Returns up to 3 options, fastest first. Each option has legs (walk / bus) with ride "
                     "and wait minutes, the road-following path as [lat, lon] points, a confidence label, "
                     "and (for depart=now) the live buses approaching each boarding stop with their "
                     "vehicle number and ETA.")
def journeys_endpoint(
        from_: str = Query(..., alias="from", description="'lat,lon' or 'stop:<stop_id>'", examples=["28.6139,77.2090"]),
        to: str = Query(..., description="'lat,lon' or 'stop:<stop_id>'", examples=["28.5355,77.2640"]),
        depart: str = Query("now", description="'now' or HH:MM in IST (24-hour)"),
        max_transfers: int = Query(2, ge=0, le=3),
        live: bool = Query(True, description="use live data (only applies when depart=now)"),
        max_options: int = Query(3, ge=1, le=5)):
    return _journeys(from_, to, depart, max_transfers, live, max_options)


@app.get("/stops")
def stops(q: str = Query(..., min_length=2), limit: int = Query(10, le=50)):
    """Find stops by name. Names that start with the text come first."""
    net = planner().net
    q = q.lower().strip()
    starts, inside = [], []
    for i, name in enumerate(net.stop_names):
        n = name.lower()
        if n.startswith(q):
            starts.append(i)
        elif q in n:
            inside.append(i)
        if len(starts) >= limit:
            break
    out = []
    for i in (starts + inside)[:limit]:
        out.append({"stop_id": int(net.stop_ids[i]), "name": net.stop_names[i],
                    "lat": float(net.stop_lat[i]), "lon": float(net.stop_lon[i]),
                    "routes": len(net.stop_routes[i])})
    return out
