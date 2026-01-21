# This program is use opencv to detect a red rope when you jump rope
# use our algorithm to count the number you jump likewise you can jump once or twice
import numpy as np
import cv2
# detect red rope
def detectRope(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # set the color threshold of red
    light_red_low = np.array([5, 110, 110])
    light_red_high = np.array([8, 255, 255])
    dark_red_low = np.array([175, 120, 120])
    dark_red_high = np.array([180, 255, 255])
    mask_light = cv2.inRange(hsv, light_red_low, light_red_high)
    mask_dark = cv2.inRange(hsv, dark_red_low, dark_red_high)
    mask = cv2.bitwise_or(mask_light, mask_dark)

    # Morphology to handle the red part you detected
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow("mask", mask)

    # find the contours of the red part and put it on another black mask
    # then use algorithm to find valid parts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rope_mask = np.zeros_like(mask, dtype=np.uint8)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        valid_contours.append(cnt)
    cv2.drawContours(rope_mask, valid_contours, -1, 255, -1)

    # compute the middle point of all the red part
    all_points = np.concatenate(valid_contours)
    M = cv2.moments(all_points)
    if M["m00"] == 0:
        return None, None
    Cx = int(M["m10"] / M["m00"])
    Cy = int(M["m01"] / M["m00"])
    return (Cx, Cy), rope_mask


class jumpRopeCounter:
    def __init__(self):
        self.count = 0
        self.rope_history = []
        self.highest_point = -1
        self.lowest_point = 2
        self.distance = 0
        self.threshold = 30
        self.state = "waiting"

    def updateCounter(self, frame):
        rope_mid_xy, _ = detectRope(frame)
        if rope_mid_xy is None:
            print("No red rope!")
            self.rope_history = []
            return None
        rope_mid_point = rope_mid_xy[1]

        self.rope_history.append(rope_mid_point)
        if len(self.rope_history) > 8:
            self.rope_history.pop(0)
        if len(self.rope_history) < 8:
            return self.count

        # get the distance between max point and min point
        self.highest_point = max(self.rope_history)
        self.lowest_point = min(self.rope_history)
        self.distance = (self.highest_point - self.lowest_point)
        current_up_trend = rope_mid_point > self.rope_history[-2]

        # update the count and reset the data in this class
        if self.distance > self.threshold:
            self.count += 1
            self.rope_history = []
            self.highest_point = -1
            self.lowest_point = 2
            self.distance = 0
            self.state = "rising"
        return self.count

    def resetCount(self):
        # reset the data
        self.count = 0
        self.rope_history = []
        self.highest_point = -1
        self.lowest_point = 2
        self.distance = 0


if __name__ == '__main__':
    # import video
    jump_rope = jumpRopeCounter()
    input_video = "twoJump.mp4"
    capture = cv2.VideoCapture(input_video)
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("Can't load the video")
            break

        rope_mid, rope_mask = detectRope(frame)
        count = jump_rope.updateCounter(frame)

        # work on special situation
        if count is None:
            cv2.putText(
                frame,
                "No Red Rope!",
                (frame.shape[1] - 200, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

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

        # adjust the size of window of the video when displaying
        scale = 0.5
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Jump Rope Counter", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            jump_rope.resetCount()
            print("Counter reset.")

    capture.release()
    cv2.destroyAllWindows()
    print(jump_rope.count)
