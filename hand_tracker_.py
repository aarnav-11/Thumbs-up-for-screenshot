
import mediapipe as mp
import cv2
import numpy as py
import time 
import threading

camera_on = True
#initialize camera
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9,10), (10,11), (11,12),      # Middle
    (0,13), (13,14), (14,15), (15,16),     # Ring
    (0,17), (17,18), (18,19), (19,20)      # Pinky
]

gestures = {
    "Fist" : [0,0,0,0,0],
    "Thumbs up00": [1,0,0,0,0],
    "Thumbs up01": [1,0,0,0,1],
    "Thumbs up11": [1,0,0,1,1],
    "Palm" : [1,1,1,1,1],
    "Peace" : [0,1,1,0,0]
}
ss_cooldown = 5
cooldown_active = False
countdown_value = None
recent_fingers = []
required_consistency = 7

def fingers_up(hand,w,h):
    landmarks = hand.landmark
    up_or_down=[]
    
    if landmarks[4].x > landmarks[2].x:
        up_or_down.append(1)
    else:
        up_or_down.append(0)

    if landmarks[8].y > landmarks[6].y:
        up_or_down.append(0)
    else:
        up_or_down.append(1)

    if landmarks[12].y > landmarks[10].y:
        up_or_down.append(0)
    else:
        up_or_down.append(1)
    
    if landmarks[16].y > landmarks[14].y:
        up_or_down.append(0)
    else:
        up_or_down.append(1)
    
    if landmarks[20].y > landmarks[18].y:
        up_or_down.append(0)
    else:
        up_or_down.append(1)
    return up_or_down

def detection(fingers):
    for name, patterns in gestures.items():
        for pattern in patterns:
            if pattern == fingers:
                print(f"Detected{name}")

def screenshot(fingers, frame):
    global cooldown_active
    if cooldown_active:
        return
    else:
        for name, pattern in gestures.items():
            if name.startswith("Thumbs up") and pattern == fingers: 
                cooldown_active = True
                threading.Thread(target=save_screenshot, args=(frame.copy(),)).start()
                

def save_screenshot(dummy_frame):
    global countdown_value, cooldown_active
    for i in reversed(range(1,6)):
        time.sleep(1)
        countdown_value = i

    ret, fresh_frame = cap.read()
    if ret:
        filename = f"thumbs_up_{int(time.time())}.png"
        cv2.imwrite(filename, fresh_frame)
        print(f"Screenshot saved: {filename}")
    else:
        print("Failed to capture fresh frame after countdown")

    cooldown_active = False
    countdown_value = None
    

if __name__ == "__main__":
    while camera_on:
        success, frame = cap.read()
        if not success:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, _ = frame.shape
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                pixel_points = []
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                for lm in hand.landmark:
                    cx,cy = int(lm.x * w), int(lm.y * h)
                    pixel_points.append((cx,cy))
                for start, end in connections:
                    cv2.line(frame, pixel_points[start], pixel_points[end], (255, 0, 0), 2)

            fingers = fingers_up(hand, w, h)

            recent_fingers.append(fingers)
            if len(recent_fingers) > required_consistency:
                recent_fingers.pop(0)

            thumbs_up_patterns = [pattern for name, pattern in gestures.items() if name.startswith("Thumbs up")]
            if (
                len(recent_fingers) == required_consistency and
                all(f == recent_fingers[0] and f in thumbs_up_patterns for f in recent_fingers)
            ):
                screenshot(fingers, frame)

        if countdown_value is not None:
            cv2.putText(frame, str(countdown_value), (900, 650), cv2.FONT_HERSHEY_DUPLEX, 11, (255, 255, 255), 25)
        
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_on = False

    cap.release()
    cv2.destroyAllWindows()

