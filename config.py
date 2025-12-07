# xmin, ymin, xmax, ymax
LEFT_ZONE = (0.0, 0.1, 0.45, 0.9)
RIGHT_ZONE = (0.55, 0.1, 1.0, 0.9)

# Thresholds for detecting right-hand swipes
RIGHT_SWIPE_DOWN_THRESHOLD = 0.02   # downward = strum down
RIGHT_SWIPE_UP_THRESHOLD   = -0.02  # upward = strum up

# chord display text (just a number for cv2 font)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX is usually 0
