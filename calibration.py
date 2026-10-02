"""
Record your own hand shape for each chord instead of relying on defaults.
Shapes are saved to chord_calibration.json next to this file.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from chords import DEFAULT_TEMPLATES
from config import CALIBRATION_FILE, CALIBRATION_FRAMES
from gestures_left import FingerCurls

DEFAULT_PATH = Path(__file__).resolve().parent / CALIBRATION_FILE


def load_templates(path: Path = DEFAULT_PATH) -> Dict[str, List[float]]:
    """Defaults, overridden by any saved calibration."""
    templates = {name: list(v) for name, v in DEFAULT_TEMPLATES.items()}
    if path.exists():
        try:
            saved = json.loads(path.read_text())
            for name, values in saved.items():
                if name in templates and len(values) == 4:
                    templates[name] = [float(v) for v in values]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[calibration] Ignoring unreadable {path.name}: {e}")
    return templates


def save_templates(templates: Dict[str, List[float]], path: Path = DEFAULT_PATH):
    path.write_text(json.dumps(templates, indent=2))


def reset_templates(path: Path = DEFAULT_PATH) -> Dict[str, List[float]]:
    if path.exists():
        path.unlink()
    return load_templates(path)


class CalibrationRecorder:
    """Averages the fretting hand's curls over several frames for one chord."""

    def __init__(self, frames: int = CALIBRATION_FRAMES):
        self.frames = frames
        self.chord: Optional[str] = None
        self._samples: List[np.ndarray] = []

    @property
    def active(self) -> bool:
        return self.chord is not None

    @property
    def progress(self) -> float:
        return len(self._samples) / self.frames

    def start(self, chord: str):
        self.chord = chord
        self._samples = []

    def feed(self, curls: FingerCurls):
        """Returns (chord, averaged curls) once enough frames are recorded."""
        if not self.active:
            return None
        self._samples.append(curls.as_array())
        if len(self._samples) < self.frames:
            return None
        chord = self.chord
        shape = [round(float(v), 3) for v in np.mean(self._samples, axis=0)]
        self.chord = None
        self._samples = []
        return chord, shape

    def cancel(self):
        self.chord = None
        self._samples = []
