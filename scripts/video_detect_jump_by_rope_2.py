import cv2
import numpy as np
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# MOG2 algorithm to filter the background
def MOG2(frame):
    fgmask = fgbg.apply(frame)
    fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    foreground = cv2.bitwise_or(frame, frame, mask = fgmask)
    return foreground
# detect red rope
def detect_rope(frame):
    foreground = MOG2(frame)
    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

    low_red1 = np.array([0, 110, 110])
    high_red1 = np.array([8, 255, 255])
    low_red2 = np.array([175, 120, 120])
    high_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, low_red1, high_red1)
    mask2 = cv2.inRange(hsv, low_red2, high_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rope_mask = np.zeros_like(mask)

    # detect the detect function output the black and white video
    #mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
    #cv2.imshow("mask", mask)

    # throw different algorithm to judge the rope  the fit cnt will append in the array
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        #compactness = area / (perimeter ** 2)
        #if compactness < 0.005:
        #    continue

        #x, y, w, h = cv2.boundingRect(cnt)
        #if min(w, h) == 0:
        #    continue
        #ratio = max(w, h) / min(w, h)
        #if ratio < 100:
        #    continue
        valid_contours.append(cnt)

    #if not valid_contours:
    #    return None, None

    # out put the video after filter
    mask = cv2.resize(rope_mask, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("mask", mask)

    # calculate the middle point
    all_points = np.concatenate(valid_contours)
    M = cv2.moments(all_points)
    if M["m00"] == 0:
        return None, None
    Cx = int(M["m10"] / M["m00"])
    Cy = int(M["m01"] / M["m00"])
    cv2.drawContours(rope_mask, valid_contours, -1, 255, -1)
    return (Cx, Cy), rope_mask

class JumpRopeCounter:
    def __init__(self):
        self.count = 0
        self.rope_history = []
        self.highest_point = -1
        self.lowest_point = 2
        self.distance = 0
        self.threshold = 30

    def updateCount(self, frame):
        rope_mid_xy, _ = detect_rope(frame)
        current_rope_mid = rope_mid_xy[1]

        # output the middle point from the function detect rope
        print(rope_mid_xy)

        self.rope_history.append(current_rope_mid)
        if len(self.rope_history) > 15:
            self.rope_history.pop(0)
        if len(self.rope_history) < 15:
            return self.count

        self.highest_point = max(self.rope_history)
        self.lowest_point = min(self.rope_history)
        self.distance = (self.highest_point - self.lowest_point)
        current_up_trend = current_rope_mid > self.rope_history[-1]

        if (self.distance > self.threshold and
            not current_up_trend and
            current_rope_mid < self.rope_history[-1] + 0.01):
            self.count += 1
            self.rope_history = []
            self.highest_point = -1
            self.lowest_point = 2
            self.distance = 0

        return self.count

    def resetCount(self):
        self.count = 0
        self.rope_history = []
        self.highest_point = -1
        self.lowest_point = 2
        self.distance = 0

if __name__ == '__main__':
    jump_rope = JumpRopeCounter()
    input_video = "twoJump.mp4"
    cap = cv2.VideoCapture(input_video)
    while cap.isOpened():
        ret, frame = cap.read()
        rope_mid, rope_mask = detect_rope(frame)
        count = jump_rope.updateCount(frame)

        if rope_mid is not None:
            cv2.circle(frame, rope_mid, 3, (0, 0, 255), -1)
            frame[rope_mask == 255] = [0, 255, 0]
            cv2.putText(
                frame,
                f"count : {count}",
                (frame.shape[1] - 200, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3)

        else:
            cv2.putText(
                frame,
                "No Rope detected",
                (frame.shape[1] - 200, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        scale = 0.8
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Jump Rope Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            jump_rope.resetCount()
            print("Counter reset.")

    cap.release()
    cv2.destroyAllWindows()
    print(jump_rope.count)