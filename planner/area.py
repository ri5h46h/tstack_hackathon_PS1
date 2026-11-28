"""
area.py  -  the geographic box the planner accepts.

DTC also runs inter-state services (Karnal, Panipat, Baraut, Chandigarh, Jaipur, Dehradun, ...),
so the box is much wider than Delhi. One place to change it.
(lat min, lat max, lon min, lon max)
"""
AREA = (26.5, 31.2, 75.5, 78.6)


def inside(lat, lon):
    return AREA[0] <= lat <= AREA[1] and AREA[2] <= lon <= AREA[3]
