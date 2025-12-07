from dataclasses import dataclass

# Mediapipe landmark indices for finger MCP joints 
FINGER_KNUCKLES = {
    "index": 5,   # INDEX_FINGER_MCP
    "middle": 9,  # MIDDLE_FINGER_MCP
    "ring": 13,   # RING_FINGER_MCP
    "pinky": 17,  # PINKY_MCP
}


@dataclass
class KnuckleHeights:
    index: float
    middle: float
    ring: float
    pinky: float


def get_knuckle_heights(hand_landmarks, normalize_to_wrist=True):
    """
    Extract knuckle Y positions for index/middle/ring/pinky.
    If normalize_to_wrist=True, we subtract the wrist.y so values are relative.
    Returns a KnuckleHeights dataclass.
    """
    wrist_y = hand_landmarks.landmark[0].y

    heights = {}
    for name, idx in FINGER_KNUCKLES.items():
        y = hand_landmarks.landmark[idx].y
        if normalize_to_wrist:
            heights[name] = y - wrist_y
        else:
            heights[name] = y

    return KnuckleHeights(**heights)


def knuckle_heights_to_str(knuckles: KnuckleHeights) -> str:
    """
    Helper to print nicely.
    """
    return (f"idx={knuckles.index:.3f}, "
            f"mid={knuckles.middle:.3f}, "
            f"ring={knuckles.ring:.3f}, "
            f"pky={knuckles.pinky:.3f}")
