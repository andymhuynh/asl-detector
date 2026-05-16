import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not found.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    sign = "No hand"
    debug = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark

            thumb_open = dist(lm[4], lm[17]) > dist(lm[3], lm[17])
            index_open = dist(lm[8], lm[0]) > dist(lm[6], lm[0])
            middle_open = dist(lm[12], lm[0]) > dist(lm[10], lm[0])
            ring_open = dist(lm[16], lm[0]) > dist(lm[14], lm[0])
            pinky_open = dist(lm[20], lm[0]) > dist(lm[18], lm[0])

            fingers = [
                int(thumb_open),
                int(index_open),
                int(middle_open),
                int(ring_open),
                int(pinky_open)
            ]

            total = sum(fingers)

            thumb_index = dist(lm[4], lm[8])
            thumb_middle = dist(lm[4], lm[12])
            index_pinky = dist(lm[8], lm[20])

            # I LOVE YOU
            if fingers == [1, 1, 0, 0, 1]:
                sign = "I LOVE YOU"

            # D = index only
            elif fingers in ([0, 1, 0, 0, 0], [1, 1, 0, 0, 0]):
                sign = "D"

            # V = index and middle
            elif fingers in ([0, 1, 1, 0, 0], [1, 1, 1, 0, 0]):
                sign = "V"

            # B = four fingers open
            elif index_open and middle_open and ring_open and pinky_open:
                sign = "B"

            # A = fist
            elif total <= 1 and thumb_index < 0.18:
                sign = "A"

            # C = curved hand
            elif (
                thumb_index > 0.10 and
                thumb_index < 0.35 and
                thumb_middle > 0.10 and
                index_pinky > 0.08 and
                total >= 2
            ):
                sign = "C"

            else:
                sign = "Unknown"

            debug = f"Fingers: {fingers}  total={total}"

    cv2.putText(frame, f"ASL Sign: {sign}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, debug, (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ASL Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
