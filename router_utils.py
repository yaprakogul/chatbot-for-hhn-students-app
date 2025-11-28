PX_TO_M = 0.15
WALK_M_PER_MIN = 84

def px_to_meters(px):
    return px * PX_TO_M

def px_to_minutes(px):
    return px_to_meters(px) / WALK_M_PER_MIN

ALIASES = {
    "mensa": ("BildungsCampus", "8"),
    "cafeteria": ("BildungsCampus", "17"),
    "auditorium": ("BildungsCampus", "15"),
    "lab f": ("TechCampus", "F"),
    "library": ("BildungsCampus", "13"),
}
