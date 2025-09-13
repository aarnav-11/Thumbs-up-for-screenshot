import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

gestures = {
  "Fist1" : [1,1,1,1],
  "Fist2" : [1,1,1,1,0],
  "Thumbs up" : [1, 0, 0, 0, 0],
  "Thumbs up with index" : [1, 0, 0, 0, 1],
  "Thumbs up with index and middle" : [1, 0, 0, 1, 1],
  "Palm" : [1,0,0,0,0],
}
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def fingers_up(hand):
    lm = hand.landmark
    index_up  = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    return index_up, middle_up

def handGesture(hand):
  up_or_down = []
  if hand.landmark[4].y < hand.landmark[2].y:
    up_or_down.append(1)
  else:
    up_or_down.append(0)
  if hand.landmark[8].y < hand.landmark[6].y:
    up_or_down.append(0)
  if hand.landmark[12].y < hand.landmark[10].y:
    up_or_down.append(0)
  else:
    up_or_down.append(1)
  if hand.landmark[16].y < hand.landmark[14].y:
    up_or_down.append(0)
  else:
    up_or_down.append(1)
  if hand.landmark[20].y < hand.landmark[18].y:
    up_or_down.append(0)
  else:
    up_or_down.append(1)
  return up_or_down

draw_color = (0, 0, 255)
brush_thickness = 6
x1, y1 = None, None

# make a persistent blank canvas (same size as webcam frames)
ret, frame0 = cap.read()
canvas = np.zeros_like(frame0)

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)  # Flip frame first
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        index_up, middle_up = fingers_up(hand)
        draw_mode = index_up and not middle_up

        tip = hand.landmark[8]
        # Use w - x to mirror the x coordinate
        x2, y2 = int(tip.x * w), int(tip.y * h)

        if draw_mode:
            if x1 is not None and y1 is not None:
                cv2.line(canvas, (x1, y1), (x2, y2), draw_color, brush_thickness)
            x1, y1 = x2, y2
        else:
            x1, y1 = None, None
          
        hand_gesture = handGesture(hand)
        print (hand_gesture)
        if hand_gesture == gestures["Palm"] :
          canvas[:] = 0

        cv2.circle(frame, (x2, y2), 6, (0, 255, 0), -1)
    
    # overlay canvas on current frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Air Draw", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
