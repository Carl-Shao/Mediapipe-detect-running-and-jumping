import cv2
import numpy as np

def detect_rope(frame):
    low_red1 = np.array([0, 110, 110])
    high_red1 = np.array([0, 255, 255])
    low_red2 = np.array([160, 110, 110])
    high_red2 = np.array([175, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, low_red1, high_red1)
    mask2 = cv2.inRange(hsv, low_red2, high_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

