from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import CURL_FULL_BEND_RAD, CURL_SMOOTHING

# Mediapipe landmark indices for each finger: (MCP, PIP, DIP, TIP)
WRIST = 0
FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_NAMES = tuple(FINGER_JOINTS)


@dataclass
class FingerCurls:
    """How bent each finger is: 0.0 = straight, 1.0 = fully curled."""
    index: float
    middle: float
    ring: float
    pinky: float

    def as_array(self) -> np.ndarray:
        return np.array([self.index, self.middle, self.ring, self.pinky])

    @classmethod
    def from_array(cls, values) -> "FingerCurls":
        return cls(*(float(v) for v in values))


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """Mediapipe landmark list -> (21, 3) numpy array of x, y, z."""
    return np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark])


def _bend_angle(a, b, c) -> float:
    """
    Angle the finger turns at joint b, in radians.
    0 when a-b-c is a straight line, pi/2 for a right-angle bend.
    """
    v1 = b - a
    v2 = c - b
    # atan2 stays precise for nearly-straight fingers, where arccos doesn't
    return float(np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2)))


def get_finger_curls(points: np.ndarray) -> FingerCurls:
    """
    Measure how curled each finger is from 21 hand landmarks.

    The old version used knuckle (MCP) heights relative to the wrist. Those
    barely move when a finger bends, but change a lot when the whole hand
    tilts or moves closer to the camera, which is why chords were unreliable.

    This sums the bend angles at the MCP, PIP and DIP joints instead. Angles
    don't depend on how big the hand is on screen or how it's rotated, so
    pass in 3D points (mediapipe's world landmarks work best).
    """
    curls = {}
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        total = (
            _bend_angle(points[WRIST], points[mcp], points[pip])
            + _bend_angle(points[mcp], points[pip], points[dip])
            + _bend_angle(points[pip], points[dip], points[tip])
        )
        curls[name] = min(total / CURL_FULL_BEND_RAD, 1.0)
    return FingerCurls(**curls)


class CurlSmoother:
    """
    Exponential moving average over frames. Mediapipe's per-frame estimates
    jitter, which made chords flicker on and off.
    """

    def __init__(self, alpha: float = CURL_SMOOTHING):
        self.alpha = alpha
        self.value: Optional[np.ndarray] = None

    def update(self, curls: FingerCurls) -> FingerCurls:
        x = curls.as_array()
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return FingerCurls.from_array(self.value)

    def reset(self):
        self.value = None


def finger_curls_to_str(curls: FingerCurls) -> str:
    """
    Helper to print nicely.
    """
    return (f"idx={curls.index:.2f}, "
            f"mid={curls.middle:.2f}, "
            f"ring={curls.ring:.2f}, "
            f"pky={curls.pinky:.2f}")
