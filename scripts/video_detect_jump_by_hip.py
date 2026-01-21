import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class JumpRopeCounter:
    def __init__(self):
        self.count = 0
        self.centers = []
        self.highest_point = -1
        self.lowest_point = 2
        self.threshold = 0.04

    def getBodyCenter(self, landmarks):
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_center = (left_hip.y + right_hip.y) / 2
        shoulder_center = (left_shoulder.y + right_shoulder.y) / 2
        center = (hip_center + shoulder_center) / 2
        return center

    def updateCount(self, landmarks):
        current_center = self.getBodyCenter(landmarks)
        self.centers.append(current_center)
        if len(self.centers) > 10:
            self.centers.pop(0)
        if len(self.centers) < 10:
            return self.count

        self.highest_point = max(self.centers)
        self.lowest_point = min(self.centers)

        current_up_trend = current_center > self.centers[-1]
        distance = self.highest_point - self.lowest_point

        if (distance > self.threshold and
        not current_up_trend and
        current_center < self.lowest_point + 0.01):
            self.count += 1
            self.centers = []
            self.highest_point = -1
            self.lowest_point = 2

        return self.count

    def reset(self):
        self.count = 0
        self.centers = []
        self.highest_point = -1
        self.lowest_point = 2

if __name__ == '__main__':
    counter = JumpRopeCounter()
    input_video = "jumpRopeVideo.mp4"
    cap = cv2.VideoCapture(input_video)

    print("Press 'q' to quit.")
    print("Press 'r' to reset.")

    with mp_pose.Pose(
            static_image_mode = False,
            model_complexity = 1,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                jump_count = counter.updateCount(results.pose_landmarks.landmark)

                cv2.putText(
                    frame,
                    f"count : {jump_count}",
                    (frame.shape[1] - 200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )
            else:
                cv2.putText(
                    frame,
                    "No Person detected",
                    (frame.shape[1] - 200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            scale = 0.8
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            cv2.imshow("Jump Rope Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                counter.reset()
                print("Counter reset.")

    cap.release()
    cv2.destroyAllWindows()
    print(counter.count)