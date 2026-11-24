"""
features.py  -  definitions shared by live serving (planner/live.py) and training
(build/train_eta.py). Both import from here, so the numbers a model learns from are
computed exactly the way they are computed when it is used.

Ideas
  expected time   what the ETA table says a bus needs to cover a stretch of its route
  observed time   what the bus really took (time between two pings)
  ratio           observed / expected over a window. 1.0 = usual, 1.3 = 30 % slower.

The ratio is measured three ways:
  network   all buses, last 10 minutes
  route     buses of the same route, last 20 minutes (leans on the network value when thin)
  own       the bus itself, over its last few ping intervals
"""
import numpy as np

# ---- ping intervals used for ratios
INTERVAL_MIN_S, INTERVAL_MAX_S = 15, 180   # ping gap that counts
TERMINAL_M = 150                           # ignore intervals near route ends (buses wait there)
VMAX = 20.0                                # m/s, spike filter
MOVE_M = 15                                # a bus has "moved" when it is this far from where it last moved
OWN_K = 6                                  # intervals in the bus's own ratio
MIN_OWN_EXP_S = 5.0                        # own ratio needs at least this much expected time

# ---- windows and priors
NET_WINDOW_MIN, ROUTE_WINDOW_MIN = 10, 20
WARM_EXPECTED_S = 3000                     # expected bus-seconds needed before the network ratio is trusted
ROUTE_PRIOR_S = 600                        # a route needs this much data to outweigh the network value
FACTOR_CLIP = (0.6, 1.8)

# ---- ETA model cases
AHEAD_M = 20                               # first stop that counts as "ahead" is > this far past the bus
HORIZONS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)   # stops ahead the model is trained on
HMAX = max(HORIZONS)

FEATURES = [
    "base", "hist", "h", "dist_rem", "spread", "frac_route", "route_len", "route_speed",
    "hour_sin", "hour_cos", "weekend",
    "f_net", "e_net", "f_route", "e_route", "own_ratio", "own_exp", "standing_s",
]


def interval_ok(dt, p0, p1, total):
    """Which ping intervals count for ratios. Arrays in, boolean array out."""
    dt, p0, p1, total = (np.asarray(a, dtype=float) for a in (dt, p0, p1, total))
    step = p1 - p0
    lo, hi = TERMINAL_M, total - TERMINAL_M
    return ((dt >= INTERVAL_MIN_S) & (dt <= INTERVAL_MAX_S)
            & (p0 >= lo) & (p0 <= hi) & (p1 >= lo) & (p1 <= hi)
            & (step >= -50) & (step <= VMAX * dt + 100))


def net_factor(obs, exp):
    """Network ratio from sums over the window. Returns 1.0 until enough data."""
    obs, exp = np.asarray(obs, dtype=float), np.asarray(exp, dtype=float)
    safe = np.where(exp > 0, exp, 1.0)
    f = np.clip(obs / safe, *FACTOR_CLIP)
    return np.where(exp >= WARM_EXPECTED_S, f, 1.0)


def route_factor(r_obs, r_exp, f_net):
    """Route ratio, mixed with the network value according to how much route data there is."""
    r_obs, r_exp, f_net = (np.asarray(a, dtype=float) for a in (r_obs, r_exp, f_net))
    safe = np.where(r_exp > 0, r_exp, 1.0)
    w = r_exp / (r_exp + ROUTE_PRIOR_S)
    f = np.clip(w * (r_obs / safe) + (1 - w) * f_net, *FACTOR_CLIP)
    return np.where(r_exp > 0, f, f_net)


def own_ratio(obs, exp):
    obs, exp = np.asarray(obs, dtype=float), np.asarray(exp, dtype=float)
    safe = np.where(exp > 0, exp, 1.0)
    return np.where(exp >= MIN_OWN_EXP_S, np.clip(obs / safe, 0.2, 5.0), np.nan)


def ist_parts(ts):
    """hour of day (float) and weekend flag (0/1) in IST from epoch seconds."""
    ts = np.asarray(ts, dtype=float)
    local = ts + 19800
    hour = (local % 86400) / 3600.0
    dow = ((local // 86400).astype(np.int64) + 3) % 7        # Monday = 0 (1 Jan 1970 was a Thursday)
    return hour, (dow >= 5).astype(float)


def build_matrix(c):
    """
    c: dict of equal-length arrays with keys
       base, hist, h, dist_rem, spread, frac_route, route_len, route_speed,
       ts (epoch seconds of the ping), f_net, e_net, f_route, e_route, own_obs, own_exp, standing_s
    Returns an (n, len(FEATURES)) float array in FEATURES order. NaN means unknown.
    """
    hour, weekend = ist_parts(c["ts"])
    cols = {
        "base": c["base"], "hist": c["hist"], "h": c["h"], "dist_rem": c["dist_rem"], "spread": c["spread"],
        "frac_route": c["frac_route"], "route_len": c["route_len"], "route_speed": c["route_speed"],
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekend": weekend,
        "f_net": c["f_net"], "e_net": c["e_net"], "f_route": c["f_route"], "e_route": c["e_route"],
        "own_ratio": own_ratio(c["own_obs"], c["own_exp"]), "own_exp": c["own_exp"],
        "standing_s": c["standing_s"],
    }
    n = len(np.asarray(c["base"]))
    return np.column_stack([np.broadcast_to(np.asarray(cols[k], dtype=float), (n,)) for k in FEATURES])


def standing_seconds(run_ids, ts, prog):
    """
    For pings sorted by (run, time): seconds since the bus last moved, where "moved" means
    it is at least MOVE_M from the place it last moved. Same rule as the live tracker.
    """
    n = len(ts)
    out = np.zeros(n)
    run_ids = np.asarray(run_ids)
    ts = np.asarray(ts, dtype=float)
    prog = np.asarray(prog, dtype=float)
    last_run, move_p, move_t = None, 0.0, 0.0
    for i in range(n):
        if run_ids[i] != last_run:
            last_run, move_p, move_t = run_ids[i], prog[i], ts[i]
        elif abs(prog[i] - move_p) >= MOVE_M:
            move_p, move_t = prog[i], ts[i]
        out[i] = ts[i] - move_t
    return out
