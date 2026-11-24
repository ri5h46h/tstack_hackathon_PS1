# Technical note: Delhi bus journey planner (PS1)

Everything here uses only our own collected bus-location data (GTFS-realtime VehiclePositions saved to CSV over several days), the GTFS static files, and the live feed.

## 1. Linking GPS points to roads and routes
- Each ping from the feed carries a `route_id`. We take that route's road shape from `shapes.txt` (one representative trip per route, the one with most stops).
- A ping is projected onto the shape. Its **progress** is the distance in metres along the route. A ping more than 100 m from the shape (150 m live) is dropped as off-route.
- Stops are placed on the shape the same way (nearest point, small back tolerance, progress never goes backwards). So every stop and every bus has one number: metres along the route.
- Bad pings are filtered: speeds above 20 m/s between pings, jumps backwards, gaps over 2 min, and a bus standing near a terminal.
- **Arrival times** at each stop come from interpolating between the two pings on either side of the stop's progress value. Runs with only coarse pings are marked "coarse" and left out of the tables.
- If a route's shape does not fit its stops (over 30% of stops more than 250 m away), we fall back to straight lines between stops for that route.

## 2. Travel time on each stretch
- A stretch is one stop-to-stop segment of one route. For each `(route, from stop, to stop, time bucket)` we store the median, p25 and p75 of observed times. Buckets: morning 6-11, midday 11-16, pm peak 16-20, night 20-6.
- Thin data is handled by blending the bucket median with the all-day median, weight `n / (n + 20)`.
- A scale factor per bucket is learned on days the table did not see (leave-one-day-out), to remove the bias from using medians.
- Missing pairs fall back in order: route speed in that bucket, route speed all day, network speed. Each journey leg reports how much of its time came from real stop-pair data (confidence high / medium / low).

## 3. Combining live and historical data
- **History** gives the expected time for every stretch (section 2).
- **Live ratio:** for every pair of consecutive pings we compare the time the bus really took with the table's expected time for that stretch. Summed over the last 10 minutes for the whole network and 20 minutes for each route (shrunk toward the network value when a route has few buses), this gives a ratio: 1.0 is usual, 1.3 is 30% slower.
- **History ETA** for a bus approaching a stop = table time from its position to the stop x route ratio, minus the seconds since its last ping.
- **AI correction (optional):** a gradient-boosted model (`build/train_eta.py`) predicts the error of the history ETA from the ratios, the bus's own recent speed, how long it has stood still, stops remaining and time of day. Training uses tables built without the day being predicted so it cannot see its own answers. The planner uses the model only if it beat the history ETA on held-out data.
- **Waiting time:** for trips starting now, the wait at a boarding stop is the time until the next live bus on that route reaches it after the rider arrives (capped at 3x the timetable wait). With no live bus it uses half the timetable headway.
- **Journey search:** RAPTOR-style rounds (up to 3 buses), walking links of up to 400 m between stops, ride times scaled by live ratios for "now" trips.

## 4. API
`GET /journeys?from=<lat,lon | stop:ID>&to=<lat,lon | stop:ID>&depart=<now | HH:MM>&max_transfers=2&live=true`

Returns options with `total_min`, `arrive`, `transfers`, `confidence`, and `legs`. Bus legs include route, stops, `ride_min`, `wait_min`, `wait_source` (live / timetable), `next_vehicle`, a road-following `path`, and `live_buses` (vehicle number, ETA, stops away, `eta_source` model / history). Interactive docs at `/docs`; the web screen at `/`. Other endpoints: `/live/stop` (stop board), `/stops` (search), `/live/status`, `/health`.

## 5. Limits
- Only our collected days are used, so rarely served stop pairs fall back to speed estimates.
- Only the longest trip pattern of each route is planned.
- Walking uses straight-line distance x 1.3, not footpaths.
- Live data is used only for trips starting now.
- Buses that have not left a terminal are not in the feed, so early waits can be underestimated.
