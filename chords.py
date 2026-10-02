# Standard tuning, low E to high e, as MIDI note numbers
STRING_MIDI = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4

# Fret per string, low E to high e. None = muted string.
CHORD_FRETS = {
    "OPEN": [0, 0, 0, 0, 0, 0],
    "C":    [None, 3, 2, 0, 1, 0],
    "G":    [3, 2, 0, 0, 0, 3],
    "D":    [None, None, 0, 2, 3, 2],
    "Am":   [None, 0, 2, 2, 1, 0],
    "Em":   [0, 2, 2, 0, 0, 0],
    "F":    [1, 3, 3, 2, 1, 1],
}

# Order matters: number keys 1-7 calibrate these in this order
CHORD_ORDER = ["OPEN", "C", "G", "D", "Am", "Em", "F"]

# Default left-hand shapes: curl of (index, middle, ring, pinky), 1 = curled.
# An air guitar can't see which string a finger is on, so each chord is a
# different set of curled fingers. C and G match the original templates
# (C = index+middle+ring, G = middle+ring+pinky). Recalibrate with keys 1-7
# to use whatever shapes feel natural to you.
DEFAULT_TEMPLATES = {
    "OPEN": [0.0, 0.0, 0.0, 0.0],   # open hand
    "C":    [1.0, 1.0, 1.0, 0.0],
    "G":    [0.0, 1.0, 1.0, 1.0],
    "D":    [1.0, 1.0, 0.0, 0.0],
    "Am":   [0.0, 1.0, 1.0, 0.0],
    "Em":   [0.0, 0.0, 1.0, 1.0],
    "F":    [1.0, 1.0, 1.0, 1.0],   # fist
}


def chord_midi_notes(chord_name):
    """MIDI note per string for a chord, None for muted strings."""
    return [
        None if fret is None else base + fret
        for base, fret in zip(STRING_MIDI, CHORD_FRETS[chord_name])
    ]
