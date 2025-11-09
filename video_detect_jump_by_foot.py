import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class JumpRopeCounts:
    def __init__(self):
        self.count = 0
        self.in_air = 0
        self.history = {"left_ankle_y": [], "right_ankle_y": [], "hip_y": []}
        self.air_threshold = 0.01
        self.land_threshold = 0.005


    def calculate_knee_ankle(self, hip, knee, ankle):
        v1 = (hip.x - knee.x, hip.y - knee.y)
        v2 = (ankle.x - hip.x, ankle.y - knee.y)
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        return np.degrees(np.arccos(dot / (mag1 * mag2 + 1e-6)))

    def getHipCenter(self, landmarks):
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        return (left_hip.y + right_hip.y) / 2

    def updateCounts(self, landmarks):
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        current_hip_y = self.getHipCenter(landmarks)
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_angle = self.calculate_knee_ankle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], left_knee, left_ankle)
        right_angle = self.calculate_knee_ankle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], right_knee, right_ankle)
        knee_bent = (90 < left_angle < 180) and (90 < right_angle < 180)

        self.history["left_ankle_y"].append(left_ankle.y)
        self.history["right_ankle_y"].append(right_ankle.y)
        self.history["hip_y"].append(current_hip_y)
        if len(self.history["left_ankle_y"]) > 3:
            for key in self.history:
                self.history[key].pop(0)

        if len(self.history["left_ankle_y"]) < 2:
            return self.count

        left_air = self.history["left_ankle_y"][-2] - left_ankle.y > self.air_threshold
        right_air = self.history["right_ankle_y"][-2] - right_ankle.y > self.air_threshold
        current_in_air = left_air and right_air

        left_land = left_ankle.y - self.history["left_ankle_y"][-2] > self.land_threshold
        right_land = right_ankle.y - self.history["right_ankle_y"][-2] > self.land_threshold
        current_land = left_land and right_land and knee_bent

        if not self.in_air and current_in_air:
            self.in_air = True
        elif self.in_air and current_land:
            self.count += 1
            self.in_air = False

        return self.count

    def reset(self):
        self.count = 0
        self.in_air = 0
        self.history = {"left_ankle_y": [], "right_ankle_y": [], "hip_y": []}

if __name__ == '__main__':

    counter = JumpRopeCounts()

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

                jump_count = counter.updateCounts(results.pose_landmarks.landmark)

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
                    "No Person Detected",
                    (frame.shape[1] - 200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            frame = cv2.resize(frame, (800, 800))
            cv2.imshow("Jump Rope Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset()
                print("Counter has been reset.")

    cap.release()
    cv2.destroyAllWindows()
    print(counter.count)