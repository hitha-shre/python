import cv2
import numpy as np
import pyautogui
import time
import math

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# OpenCV camera config
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_x, prev_y = 0, 0
smooth_factor = 5

# Gesture timers
last_click_time = 0
last_gesture_time = 0
click_delay = 1

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.dist(start, end)
            b = math.dist(start, far)
            c = math.dist(end, far)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-5)) * 57

            if angle <= 90:
                finger_count += 1
                cv2.circle(drawing, far, 5, [0, 0, 255], -1)

        return finger_count + 1
    return 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    drawing = frame.copy()

    # Convert to HSV & mask skin color range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise reduction
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area > 5000:
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)

            # Count fingers
            finger_count = count_fingers(contour, drawing)

            # Bounding box for palm center
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

            # Map hand position to screen
            screen_x = np.interp(cx, (0, 640), (0, screen_w))
            screen_y = np.interp(cy, (0, 480), (0, screen_h))

            curr_x = prev_x + (screen_x - prev_x) / smooth_factor
            curr_y = prev_y + (screen_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            now = time.time()

            # === Gesture actions ===
            if finger_count == 1 and now - last_click_time > click_delay:
                pyautogui.click()
                print("Left Click")
                last_click_time = now

            elif finger_count == 2 and now - last_click_time > click_delay:
                pyautogui.rightClick()
                print("Right Click")
                last_click_time = now

            elif finger_count == 3 and now - last_gesture_time > click_delay:
                pyautogui.press("space")
                print("Play/Pause")
                last_gesture_time = now

            elif finger_count == 4 and now - last_gesture_time > click_delay:
                pyautogui.press("right")
                print("Forward")
                last_gesture_time = now

            elif finger_count == 5 and now - last_gesture_time > click_delay:
                pyautogui.press("left")
                print("Backward")
                last_gesture_time = now

            cv2.putText(drawing, f'Fingers: {finger_count}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Control (No MediaPipe)", drawing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
