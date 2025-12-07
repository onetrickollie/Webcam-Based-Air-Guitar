# chord_detection.py

def detect_chord(knuckles, templates, tolerance=0.015, debug=False):
    """
    Compare knuckle positions to chord templates.
    Uses average error over only the fingers defined in each template.
    Returns the best chord name or None.
    """
    best_chord = None
    best_error = 999.0

    for chord_name, target in templates.items():
        error = 0.0
        count = 0

        if "index" in target:
            error += abs(knuckles.index - target["index"])
            count += 1
        if "middle" in target:
            error += abs(knuckles.middle - target["middle"])
            count += 1
        if "ring" in target:
            error += abs(knuckles.ring - target["ring"])
            count += 1
        if "pinky" in target:
            error += abs(knuckles.pinky - target["pinky"])
            count += 1

        if count == 0:
            continue

        avg_error = error / count

        if debug:
            print(f"[DEBUG] {chord_name} avg_error={avg_error:.4f}")

        if avg_error < best_error:
            best_error = avg_error
            best_chord = chord_name

    if debug:
        print(f"[DEBUG] best={best_chord}, best_error={best_error:.4f}")

    return best_chord if best_error < tolerance else None
