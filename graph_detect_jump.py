import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def is_jumping(landmarks):
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

    if (left_ankle_y > left_knee_y - 0.05 and right_ankle_y > right_knee_y - 0.05) and \
    (left_foot_y > left_knee_y - 0.05 and right_foot_y > right_foot_y - 0.05):
        return True
    return False

input_path = "jump.jpg"
image = cv2.imread(input_path)
if image is None:
    print("Could not load image")
    exit()

with mp_pose.Pose(static_image_mode = True, model_complexity = 2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        if is_jumping(results.pose_landmarks.landmark):
            cv2.putText(image, "Jumping!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Not Jumping", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No pose landmarks")

    resize_image = cv2.resize(image, (800, 800))

    cv2.imwrite("jump_result.jpg", resize_image)
    cv2.imshow("Image", resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()