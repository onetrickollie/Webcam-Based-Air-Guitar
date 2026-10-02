import argparse
import time

import cv2
import mediapipe as mp

from audio_player import AudioPlayer
from calibration import CalibrationRecorder, load_templates, reset_templates, save_templates
from chord_detection import ChordTracker
from chords import CHORD_ORDER
from config import FONT, LEFT_ZONE, RIGHT_ZONE
from gestures_left import (
    FINGER_NAMES,
    CurlSmoother,
    finger_curls_to_str,
    get_finger_curls,
    landmarks_to_array,
)
from gestures_right import StrumDetector
from zones import hand_in_zone, palm_center

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GREEN = (0, 255, 0)
BLUE = (255, 120, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)

FRET_GRACE_SEC = 0.25    # keep the chord through brief tracking dropouts
STRUM_FLASH_SEC = 0.2
SHORT_NAMES = {"index": "idx", "middle": "mid", "ring": "rng", "pinky": "pky"}


def put_text(frame, text, org, font, scale, color, thickness):
    """cv2.putText with a dark outline so it's readable over any background."""
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def zone_px(zone, w, h):
    return int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)


def draw_zone(frame, zone, label, color, active):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = zone_px(zone, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if active else 1)
    put_text(frame, label, (x1 + 5, y1 - 10), FONT, 0.5, color, 1)


def draw_curl_bars(frame, curls, x, y):
    for i, name in enumerate(FINGER_NAMES):
        value = getattr(curls, name)
        top = y + i * 18
        put_text(frame, SHORT_NAMES[name], (x, top + 11), FONT, 0.4, WHITE, 1)
        cv2.rectangle(frame, (x + 35, top), (x + 135, top + 12), (40, 40, 40), -1)
        cv2.rectangle(frame, (x + 35, top), (x + 35 + int(100 * value), top + 12), GREEN, -1)
        cv2.rectangle(frame, (x + 35, top), (x + 135, top + 12), GRAY, 1)


def draw_help(frame, show_help):
    h, w, _ = frame.shape
    if not show_help:
        put_text(frame, "h: help", (10, h - 10), FONT, 0.45, GRAY, 1)
        return
    keys = "  ".join(f"{i + 1}:{name}" for i, name in enumerate(CHORD_ORDER))
    lines = [
        "Hold a chord shape with your left hand, then press its number to calibrate:",
        keys,
        "r: reset calibration   d: debug output   h: hide help   ESC/q: quit",
    ]
    # dim the strip behind the help so it doesn't fight with the zones
    top = h - 20 * len(lines) - 12
    strip = frame[top:h]
    strip[:] = (strip * 0.35).astype(strip.dtype)
    for i, line in enumerate(lines):
        put_text(frame, line, (10, h - 12 - 20 * (len(lines) - 1 - i)), FONT, 0.45, WHITE, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam air guitar")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--no-sound", action="store_true", help="run without audio output")
    parser.add_argument("--debug", action="store_true", help="print curls and chord distances")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open the camera. On macOS, allow camera access for your "
              "terminal app in System Settings > Privacy & Security > Camera.")
        return

    templates = load_templates()
    chord_tracker = ChordTracker(templates)
    smoother = CurlSmoother()
    strum_detector = StrumDetector()
    recorder = CalibrationRecorder()
    audio = AudioPlayer(enable_output=not args.no_sound)

    debug = args.debug
    show_help = True
    curls = None
    last_fret_seen = 0.0
    last_strum = None        # (direction, time)
    status_msg = ("", 0.0)   # (text, expires at)

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()

            # Flip horizontally so it feels like a mirror
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Decide which hand is which by where it is on screen, not by
            # mediapipe's Left/Right label: the label sometimes flips between
            # frames, which made the old version drop the fretting hand.
            fret_hand = strum_hand = None
            if results.multi_hand_landmarks:
                world = results.multi_hand_world_landmarks or results.multi_hand_landmarks
                for screen_lm, world_lm in zip(results.multi_hand_landmarks, world):
                    mp_drawing.draw_landmarks(frame, screen_lm, mp_hands.HAND_CONNECTIONS)
                    points = landmarks_to_array(screen_lm)
                    if fret_hand is None and hand_in_zone(points, LEFT_ZONE):
                        fret_hand = landmarks_to_array(world_lm)
                    elif strum_hand is None and hand_in_zone(points, RIGHT_ZONE):
                        strum_hand = points

            # --- left hand: chord shape ---
            if fret_hand is not None:
                last_fret_seen = now
                curls = smoother.update(get_finger_curls(fret_hand))
                chord_tracker.update(curls, debug=debug)
                if debug:
                    print("Left hand curls:", finger_curls_to_str(curls))

                done = recorder.feed(curls)
                if done:
                    chord, shape = done
                    templates[chord] = shape
                    save_templates(templates)
                    status_msg = (f"Saved {chord} shape", now + 2.0)
                    print(f"[calibration] {chord} = {shape}")
            elif now - last_fret_seen > FRET_GRACE_SEC:
                smoother.reset()
                chord_tracker.reset()
                curls = None

            # --- right hand: strumming ---
            if strum_hand is not None:
                _, y = palm_center(strum_hand)
                strum = strum_detector.update(y, now)
                if strum:
                    direction, strength = strum
                    audio.strum(chord_tracker.current, direction, strength)
                    last_strum = (direction, now)
                    if debug:
                        print(f">>> STRUM {direction.upper()} {chord_tracker.current} "
                              f"strength={strength:.2f}")
            else:
                strum_detector.reset()

            # --- drawing ---
            draw_zone(frame, LEFT_ZONE, "FRET (left hand)", GREEN, fret_hand is not None)
            draw_zone(frame, RIGHT_ZONE, "STRUM (right hand)", BLUE, strum_hand is not None)

            chord_color = GREEN if fret_hand is not None else GRAY
            put_text(frame, chord_tracker.current, (10, 50), FONT, 1.6, chord_color, 3)
            if curls is not None:
                draw_curl_bars(frame, curls, 10, 70)

            if last_strum and now - last_strum[1] < STRUM_FLASH_SEC:
                arrow = "STRUM DOWN" if last_strum[0] == "down" else "STRUM UP"
                h, w, _ = frame.shape
                put_text(frame, arrow, (int(w * 0.6), 50), FONT, 1.0, RED, 3)

            if recorder.active:
                put_text(frame, f"Hold {recorder.chord} shape... {int(recorder.progress * 100)}%",
                            (10, 160), FONT, 0.7, YELLOW, 2)
            elif now < status_msg[1]:
                put_text(frame, status_msg[0], (10, 160), FONT, 0.7, YELLOW, 2)

            if args.no_sound or audio.stream is None:
                put_text(frame, "(sound off)", (10, 185), FONT, 0.5, GRAY, 1)

            draw_help(frame, show_help)
            cv2.imshow("Air Guitar", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC
                break
            if ord("1") <= key < ord("1") + len(CHORD_ORDER):
                chord = CHORD_ORDER[key - ord("1")]
                if fret_hand is None:
                    status_msg = ("Put your left hand in the FRET zone first", now + 2.0)
                else:
                    recorder.start(chord)
            elif key == ord("r"):
                recorder.cancel()
                templates = reset_templates()
                chord_tracker.templates = templates
                status_msg = ("Calibration reset to defaults", now + 2.0)
            elif key == ord("d"):
                debug = not debug
            elif key == ord("h"):
                show_help = not show_help

    audio.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
