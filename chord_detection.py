from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from config import CHORD_MAX_DISTANCE, CHORD_STABLE_FRAMES
from gestures_left import FingerCurls


def detect_chord(curls: FingerCurls, templates: Dict[str, Sequence[float]],
                 max_distance=CHORD_MAX_DISTANCE, debug=False) -> Tuple[Optional[str], float]:
    """
    Compare finger curls to chord templates.
    Uses RMS difference over the four fingers.
    Returns (best chord name or None, its distance).
    """
    x = curls.as_array()
    best_chord = None
    best_dist = float("inf")

    for chord_name, target in templates.items():
        dist = float(np.sqrt(np.mean((x - np.asarray(target)) ** 2)))

        if debug:
            print(f"[DEBUG] {chord_name} dist={dist:.3f}")

        if dist < best_dist:
            best_dist = dist
            best_chord = chord_name

    if debug:
        print(f"[DEBUG] best={best_chord}, best_dist={best_dist:.3f}")

    if best_dist > max_distance:
        return None, best_dist
    return best_chord, best_dist


class ChordTracker:
    """
    Holds the current chord and only switches after a new chord has been the
    best match for several frames in a row. A shape that matches nothing
    keeps the current chord, so passing between shapes doesn't drop out.
    """

    def __init__(self, templates, stable_frames=CHORD_STABLE_FRAMES,
                 max_distance=CHORD_MAX_DISTANCE, default="OPEN"):
        self.templates = templates
        self.stable_frames = stable_frames
        self.max_distance = max_distance
        self.default = default
        self.current = default
        self._candidate = None
        self._count = 0

    def update(self, curls: FingerCurls, debug=False) -> str:
        chord, _ = detect_chord(curls, self.templates, self.max_distance, debug)

        if chord is None or chord == self.current:
            self._candidate = None
            self._count = 0
            return self.current

        if chord == self._candidate:
            self._count += 1
        else:
            self._candidate = chord
            self._count = 1

        if self._count >= self.stable_frames:
            self.current = chord
            self._candidate = None
            self._count = 0

        return self.current

    def reset(self):
        """Fretting hand left the zone: nothing pressed means open strings."""
        self.current = self.default
        self._candidate = None
        self._count = 0
