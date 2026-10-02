import numpy as np

# Wrist + the four finger knuckles: their average is the middle of the palm
PALM_POINTS = [0, 5, 9, 13, 17]


def palm_center(points: np.ndarray):
    """(x, y) of the palm center from a (21, 3) landmark array."""
    center = points[PALM_POINTS, :2].mean(axis=0)
    return float(center[0]), float(center[1])


def point_in_zone(x, y, zone):
    x_min, y_min, x_max, y_max = zone
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def hand_in_zone(points: np.ndarray, zone):
    """
    Uses the palm center instead of the wrist: the wrist sits at the edge of
    the hand and was often outside the zone while the hand itself was in it.
    """
    return point_in_zone(*palm_center(points), zone)
