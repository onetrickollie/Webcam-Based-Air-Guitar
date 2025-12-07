# main.py

import cv2
import mediapipe as mp

from config import (
    LEFT_ZONE,
    RIGHT_ZONE,
    RIGHT_SWIPE_DOWN_THRESHOLD,
    RIGHT_SWIPE_UP_THRESHOLD,
)
from zones import hand_in_zone
from gestures_left import get_knuckle_heights, knuckle_heights_to_str
from gestures_right import RightHandSwipeDetector

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)

    right_swipe_detector = RightHandSwipeDetector(
        down_threshold=RIGHT_SWIPE_DOWN_THRESHOLD,
        up_threshold=RIGHT_SWIPE_UP_THRESHOLD
    )

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally so it feels like a mirror
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Draw left/right zones for debugging
            lx1, ly1, lx2, ly2 = int(LEFT_ZONE[0]*w), int(LEFT_ZONE[1]*h), int(LEFT_ZONE[2]*w), int(LEFT_ZONE[3]*h)
            rx1, ry1, rx2, ry2 = int(RIGHT_ZONE[0]*w), int(RIGHT_ZONE[1]*h), int(RIGHT_ZONE[2]*w), int(RIGHT_ZONE[3]*h)

            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            cv2.putText(frame, "LEFT HAND AREA", (lx1+5, ly1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(frame, "RIGHT HAND AREA", (rx1+5, ry1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Run mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" or "Right"

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    if label == "Left":
                        # Only care if left hand is inside LEFT_ZONE
                        if hand_in_zone(hand_landmarks, LEFT_ZONE):
                            cv2.putText(frame, "LEFT IN ZONE", (lx1+10, ly1+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # STEP 1: get knuckle heights and show them
                            knuckles = get_knuckle_heights(hand_landmarks)
                            text = knuckle_heights_to_str(knuckles)

                            # Show on screen and print in console
                            cv2.putText(frame, text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            print("Left hand knuckles:", text)

                        else:
                            cv2.putText(frame, "LEFT OUTSIDE ZONE", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    elif label == "Right":
                        # Only care if right hand is inside RIGHT_ZONE
                        wrist = hand_landmarks.landmark[0]
                        wrist_y = wrist.y

                        if hand_in_zone(hand_landmarks, RIGHT_ZONE):
                            cv2.putText(frame, "RIGHT IN ZONE", (rx1+10, ry1+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            # Detect swipe direction
                            swipe = right_swipe_detector.update(wrist_y)
                            if swipe == "down":
                                cv2.putText(frame, "STRUM DOWN", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                print(">>> STRUM DOWN (all 6 strings)")
                                # TODO: call audio_player.play_strum_down(current_chord)

                            elif swipe == "up":
                                cv2.putText(frame, "STRUM UP", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                print(">>> STRUM UP (all 6 strings)")
                                # TODO: call audio_player.play_strum_up(current_chord)
                        else:
                            cv2.putText(frame, "RIGHT OUTSIDE ZONE", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Air Guitar", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
