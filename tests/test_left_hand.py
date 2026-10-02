import numpy as np
import pytest

from chord_detection import ChordTracker, detect_chord
from chords import CHORD_ORDER, DEFAULT_TEMPLATES
from conftest import curled, make_hand, rotation
from gestures_left import CurlSmoother, FingerCurls, get_finger_curls


def test_open_hand_is_not_curled():
    curls = get_finger_curls(make_hand({}))
    assert np.allclose(curls.as_array(), 0.0, atol=1e-6)


def test_fist_is_fully_curled():
    hand = make_hand({f: curled() for f in ("index", "middle", "ring", "pinky")})
    assert np.allclose(get_finger_curls(hand).as_array(), 1.0)


def test_only_bent_finger_reads_as_curled():
    curls = get_finger_curls(make_hand({"middle": curled(0.6)}))
    assert curls.middle == pytest.approx(0.6)
    assert curls.index == pytest.approx(0.0, abs=1e-6)
    assert curls.ring == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("angles,scale,shift", [
    ((0.3, -0.5, 1.2), 1.0, (0, 0, 0)),
    ((1.1, 0.4, -2.0), 2.5, (0.3, -0.1, 0.8)),   # closer to camera
    ((0.0, 0.0, np.pi / 2), 0.4, (5, 5, 5)),     # sideways and far away
])
def test_curls_ignore_hand_size_position_and_rotation(angles, scale, shift):
    """The old knuckle-height method failed exactly this: tilting the hand
    or moving it closer changed the reading."""
    hand = make_hand({"index": curled(0.8), "ring": curled(0.4)})
    moved = (hand @ rotation(*angles).T) * scale + np.array(shift)
    assert np.allclose(get_finger_curls(moved).as_array(),
                       get_finger_curls(hand).as_array(), atol=1e-9)


def test_smoother_reduces_jitter():
    smoother = CurlSmoother(alpha=0.5)
    rng = np.random.default_rng(0)
    raw = [FingerCurls.from_array(0.5 + rng.normal(0, 0.1, 4)) for _ in range(200)]
    smoothed = np.array([smoother.update(c).as_array() for c in raw])
    raw_arr = np.array([c.as_array() for c in raw])
    assert smoothed[20:].std() < raw_arr[20:].std() * 0.75


def test_default_templates_are_distinct_and_detected():
    for name in CHORD_ORDER:
        chord, dist = detect_chord(FingerCurls.from_array(DEFAULT_TEMPLATES[name]),
                                   DEFAULT_TEMPLATES)
        assert chord == name and dist == pytest.approx(0)


def test_partial_curl_still_matches_nearest_chord():
    # a real "C" hand rarely curls fingers all the way
    chord, _ = detect_chord(FingerCurls(0.75, 0.8, 0.7, 0.15), DEFAULT_TEMPLATES)
    assert chord == "C"


def test_ambiguous_shape_matches_nothing():
    chord, _ = detect_chord(FingerCurls(0.5, 0.5, 0.5, 0.5), DEFAULT_TEMPLATES,
                            max_distance=0.35)
    assert chord is None


def test_tracker_ignores_single_frame_flicker():
    tracker = ChordTracker(DEFAULT_TEMPLATES, stable_frames=3)
    c = FingerCurls.from_array(DEFAULT_TEMPLATES["C"])
    g = FingerCurls.from_array(DEFAULT_TEMPLATES["G"])
    for _ in range(3):
        tracker.update(c)
    assert tracker.current == "C"
    tracker.update(g)            # one bad frame
    tracker.update(c)
    assert tracker.current == "C"
    for _ in range(3):
        tracker.update(g)
    assert tracker.current == "G"


def test_tracker_holds_chord_through_unrecognized_shapes():
    tracker = ChordTracker(DEFAULT_TEMPLATES, stable_frames=1)
    tracker.update(FingerCurls.from_array(DEFAULT_TEMPLATES["D"]))
    tracker.update(FingerCurls(0.5, 0.5, 0.5, 0.5))  # mid-transition
    assert tracker.current == "D"
    tracker.reset()
    assert tracker.current == "OPEN"
