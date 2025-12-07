from typing import Optional

class RightHandSwipeDetector:
    """
    Tracks wrist's Y position and detects up/dpwn swipes
    """
    def __init__(self, down_threshold = 0.02, up_threshold = 0.02):
        self.prev_y = Optional[float] = None
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold

    def update(self, wrist_y: float):
        """
        returns up and down or none to determine how the note should sound
        """
        swipe = None

        if self.prev_y is not None:
            vel = self.prev_y - wrist_y #positive = down
            if vel > self.down_threshold:
                swipe = "down"
            elif vel < self.up_threshold:
                swipe = "up"
            
        self.prev_y = wrist_y
        return swipe