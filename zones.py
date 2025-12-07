def hand_in_zone(hand_landmarks,zone):
    x_min,y_min,x_max,y_max = zone
    wrist = hand_landmarks.landmark[0]
    x = wrist.x
    y = wrist.y

    return (x_min <= x <= x_max) and (y_min <= y <= y_max)