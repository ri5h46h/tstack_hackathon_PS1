# Pipeline

## 1. Building the learned data (offline)

```mermaid
flowchart TD
    A["Delhi live feeds<br/>Transport Stack (main), OTD (backup)"] --> B["collector/otd_collector.py<br/>fetch, decode, validate, save"]
    B --> C[("data/clean/DAY.csv<br/>one row per bus ping")]
    B --> H[("data/health/<br/>feed reliability log")]
    G[("gtfs/<br/>shapes, stops, trips")] --> D
    C --> D["build/build_arrivals.py<br/>project pings on road shape<br/>drop noisy pings<br/>cut into runs<br/>find arrival time at each stop"]
    D --> E[("data/arrivals/<br/>arrivals, pings, runs")]
    E --> F["build/build_eta_bucketed.py<br/>median time per stretch and time bucket<br/>blend with all-day<br/>learn scale on unseen days"]
    E --> M["build/train_eta.py<br/>learn correction to the history ETA<br/>cross-fitted tables"]
    E --> V["eval/validate_trips.py<br/>real trip time vs ours vs GTFS timetable"]
    F --> T[("data/eta_bucketed.csv.gz<br/>eta_bucketed_info.json")]
    M --> MD[("models/eta_model.joblib")]
    V --> R["VALIDATION.md<br/>eval/validation/"]
    T --> V
```

## 2. Answering a request (online)

```mermaid
flowchart TD
    U["User<br/>web page or GET /journeys"] --> API["api/app.py (FastAPI)"]
    API --> S["planner/search.py<br/>1 find stops near start and end<br/>2 RAPTOR rounds, up to 3 buses<br/>3 describe options"]

    GT[("gtfs/")] --> N["planner/network.py<br/>stops, routes, road shapes"]
    N --> S
    TB[("data/eta_bucketed.*")] --> S

    LF["Live feed every 15 s"] --> LV["planner/live.py<br/>LiveFeed, LiveModel"]
    LV --> RATIO["traffic ratio<br/>network 10 min, route 20 min"]
    LV --> WAIT["WaitModel<br/>next live bus at boarding stop"]
    LV --> ETA["ETA of approaching buses<br/>table time x ratio"]
    MD[("models/eta_model.joblib")] -.->|optional correction| ETA
    RATIO --> S
    WAIT --> S
    ETA --> S

    S --> OUT["JSON: walk, bus legs, road path,<br/>wait source, live buses with vehicle number,<br/>confidence, usual time at a quieter hour"]
    OUT --> UI["Web page<br/>map, option cards, stop board, live pill"]
```

## 3. Shared code

`planner/features.py` defines the live traffic ratios and ETA model inputs once. Both `build/train_eta.py` (training) and `planner/live.py` (serving) use it, so the model sees the same numbers in training and in use.
