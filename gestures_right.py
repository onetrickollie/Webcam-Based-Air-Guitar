from typing import Optional, Tuple

from config import (
    STRUM_MAX_SPEED,
    STRUM_REARM_SPEED,
    STRUM_SMOOTHING,
    STRUM_TRIGGER_SPEED,
)


class StrumDetector:
    """
    Tracks the strumming hand's Y position and fires one event per strum.

    Fixes from the old RightHandSwipeDetector:
      - Image y grows downward, so moving the hand down makes y bigger. The
        old code computed prev_y - y and called that "down", which was
        backwards.
      - It fired on every frame of a swipe. Now a strum fires once, and the
        same direction can't fire again until the hand slows down (or
        reverses, for fast down-up strumming).
      - Velocity is per second instead of per frame, so the feel doesn't
        change with the camera's frame rate.
    """

    def __init__(self, trigger_speed=STRUM_TRIGGER_SPEED, rearm_speed=STRUM_REARM_SPEED,
                 smoothing=STRUM_SMOOTHING, max_speed=STRUM_MAX_SPEED):
        self.trigger_speed = trigger_speed
        self.rearm_speed = rearm_speed
        self.smoothing = smoothing
        self.max_speed = max_speed
        self.reset()

    def reset(self):
        self.prev_y: Optional[float] = None
        self.prev_t: Optional[float] = None
        self.velocity = 0.0
        self.last_direction: Optional[str] = None
        self.armed = True

    def update(self, y: float, t: float) -> Optional[Tuple[str, float]]:
        """
        y: hand height as a fraction of the frame (0 = top)
        t: timestamp in seconds
        Returns ("down" | "up", strength 0..1) when a strum fires, else None.
        """
        if self.prev_y is None or t <= self.prev_t:
            self.prev_y, self.prev_t = y, t
            return None

        raw = (y - self.prev_y) / (t - self.prev_t)  # positive = moving down
        self.prev_y, self.prev_t = y, t
        self.velocity = self.smoothing * raw + (1 - self.smoothing) * self.velocity

        speed = abs(self.velocity)
        if speed < self.rearm_speed:
            self.armed = True
        if speed < self.trigger_speed:
            return None

        direction = "down" if self.velocity > 0 else "up"
        if not self.armed and direction == self.last_direction:
            return None

        self.last_direction = direction
        self.armed = False
        strength = min(speed / self.max_speed, 1.0)
        return direction, strength
