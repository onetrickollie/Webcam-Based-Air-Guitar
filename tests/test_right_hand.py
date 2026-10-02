import numpy as np

from gestures_right import StrumDetector

FPS = 30


def run(detector, ys, fps=FPS):
    """Feed y positions at a fixed frame rate, return the strums fired."""
    events = []
    for i, y in enumerate(ys):
        strum = detector.update(y, i / fps)
        if strum:
            events.append(strum[0])
    return events


def swipe(start, end, frames):
    return list(np.linspace(start, end, frames))


def hold(y, frames):
    return [y] * frames


def test_hand_moving_down_screen_is_a_down_strum():
    # y grows downward in image coordinates; the old code had this backwards
    assert run(StrumDetector(), hold(0.4, 5) + swipe(0.4, 0.8, 6)) == ["down"]


def test_hand_moving_up_screen_is_an_up_strum():
    assert run(StrumDetector(), hold(0.8, 5) + swipe(0.8, 0.4, 6)) == ["up"]


def test_one_swipe_fires_once_not_every_frame():
    events = run(StrumDetector(), hold(0.3, 5) + swipe(0.3, 0.9, 15))
    assert events == ["down"]


def test_fast_down_up_strumming_alternates():
    ys = hold(0.5, 3)
    for _ in range(4):
        ys += swipe(0.4, 0.7, 5)[1:] + swipe(0.7, 0.4, 5)[1:]
    events = run(StrumDetector(), ys)
    assert events == ["down", "up"] * 4


def test_two_down_strums_need_a_pause_between():
    ys = hold(0.4, 3) + swipe(0.4, 0.7, 5) + hold(0.7, 5) + swipe(0.7, 0.95, 5)
    assert run(StrumDetector(), ys) == ["down", "down"]


def test_jitter_does_not_strum():
    rng = np.random.default_rng(1)
    ys = list(0.6 + rng.normal(0, 0.004, 120))
    assert run(StrumDetector(), ys) == []


def test_same_motion_strums_at_any_frame_rate():
    for fps in (15, 30, 60):
        frames = int(0.15 * fps) + 1  # 0.3 screen heights in 150 ms
        ys = hold(0.4, 3) + swipe(0.4, 0.7, frames)
        assert run(StrumDetector(), ys, fps) == ["down"], fps


def test_faster_strum_is_stronger():
    def strength(frames):
        d = StrumDetector()
        for i, y in enumerate(hold(0.3, 3) + swipe(0.3, 0.8, frames)):
            s = d.update(y, i / FPS)
            if s:
                return s[1]
    assert strength(4) > strength(10)
