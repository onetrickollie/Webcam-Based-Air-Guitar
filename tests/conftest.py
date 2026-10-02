import sys
from pathlib import Path

import numpy as np

# Let tests import the project modules (main.py lives at the repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gestures_left import FINGER_JOINTS  # noqa: E402

MCP_X = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
BONES = (0.045, 0.025, 0.02)  # MCP->PIP, PIP->DIP, DIP->TIP in meters


def make_hand(bends):
    """
    Synthetic 21-point hand with the wrist at the origin.

    bends maps finger -> (mcp, pip, dip) bend angles in radians; fingers not
    listed are straight. Each finger starts out pointing along its
    wrist->knuckle direction and curls toward -z (into the palm).
    """
    pts = np.zeros((21, 3))
    for i in range(1, 5):  # thumb: plausible but unused for curls
        pts[i] = [-0.04 - 0.01 * i, 0.02 * i, 0.0]

    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        base = np.array([MCP_X[name], 0.09, 0.0])
        pts[mcp] = base
        forward = base / np.linalg.norm(base)
        down = np.array([0.0, 0.0, -1.0])

        heading = 0.0
        pos = base
        for joint, bone, bend in zip((pip, dip, tip), BONES, bends.get(name, (0, 0, 0))):
            heading += bend
            pos = pos + bone * (np.cos(heading) * forward + np.sin(heading) * down)
            pts[joint] = pos
    return pts


def curled(amount=1.0):
    """Bend angles for a finger curled by `amount` of a full curl (3 rad total)."""
    return (amount, amount, amount)


def rotation(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx
