import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(hip, knee, ankle):
    v1 = (hip.x - knee.x, hip.y - knee.y)
    v2 = (ankle.x - knee.x, ankle.y - knee.y)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    return np.degrees(np.arccos(np.dot(v1, v2) / (mag1 * mag2 + 1e-6)))

def is_running(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    knee_bent = (120 < left_angle < 180) or (120 < right_angle < 180)

    if (abs(left_ankle.x - right_ankle.x) > 0.15) and knee_bent and \
            (abs(left_ankle.y - right_ankle.y) > 0.07):
        return True

    return False

input_path = "run.jpg"
image = cv2.imread(input_path)
if image is None:
    print("Failed to load image")
    exit()

with mp_pose.Pose(
    static_image_mode = True,
    model_complexity = 2
) as pose:
    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        if is_running(result.pose_landmarks.landmark):
            cv2.putText(image, "Running!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(image, "Not running!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        print("No pose landmarks")

    scale = 0.8
    resize_image = cv2.resize(image, (0, 0), fx = scale, fy = scale)

    cv2.imwrite("run_result.jpg", resize_image)
    cv2.imshow("Image", resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()