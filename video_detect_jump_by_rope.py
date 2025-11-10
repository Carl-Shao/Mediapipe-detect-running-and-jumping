import cv2
import numpy as np

def detect_rope(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_red1 = np.array([0, 110, 110])
    high_red1 = np.array([8, 255, 255])
    low_red2 = np.array([175, 120, 120])
    high_red2 = np.array([210, 255, 255])
    mask1 = cv2.inRange(hsv, low_red1, high_red1)
    mask2 = cv2.inRange(hsv, low_red2, high_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rope_mask = np.zeros_like(mask)
    mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("mask", mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        compactness = area / (perimeter ** 2)
        if compactness < 0.005:
            continue

        #x, y, w, h = cv2.boundingRect(cnt)
        #if min(w, h) == 0:
        #    continue
        #ratio = max(w, h) / min(w, h)
        #if ratio < 100:
        #    continue
        cv2.drawContours(rope_mask, [cnt], -1, 255, -1)


        mask = cv2.resize(rope_mask, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("mask", mask)

        point = rope_mask.reshape(-1, 2)
        mid_x = int(np.mean(point[:, 0]))
        mid_y = int(np.mean(point[:, 1]))
        rope_mid_position = (mid_x, mid_y)
        return rope_mid_position, rope_mask
    return (None, None), None

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

        rope_mid, _ = detect_rope(frame)
        _, rope_mask = detect_rope(frame)
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