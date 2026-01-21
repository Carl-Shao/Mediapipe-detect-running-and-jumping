#Mediapipe detect dancing key point program
#This program allows user to get the important data when you provide a video about dancing
#and then  give the key point and the angle of joint to the user's robots
import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class MotionDetector:
    def __init__(self):
        self.joint_mapping = {}
        self.angle_mapping = {}
        self.create_joint_mapping()

    def create_joint_mapping(self):
        #mapping the people's joint to robot
        for landmark_enum in mp_pose.PoseLandmark:
            human_joint = f"human_{landmark_enum.name.lower()}"
            robot_joint = f"robot_{landmark_enum.name.lower()}"
            self.joint_mapping[human_joint] = robot_joint
        return self.joint_mapping

    def detect_all_keypoints(self, landmarks):
        #use mediapipe to get the 33 key points in user's video
        for landmark_enum in mp_pose.PoseLandmark:
            #generate point name
            #for example human_left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
            joint_name = f"human_{landmark_enum.name.lower()}"
            setattr(self, joint_name, landmarks[landmark_enum])

    def calculate_angles(self, joint_a, joint_b, joint_c):
        #calculate the angles among three points
        #get x y z of three points
        point_a = np.array([joint_a.x, joint_a.y, joint_a.z])
        point_b = np.array([joint_b.x, joint_b.y, joint_b.z])
        point_c = np.array([joint_c.x, joint_c.y, joint_c.z])

        #use formular arccos theta = (dot process / mod_a * mod_b)
        vector_ba = point_a - point_b
        vector_bc = point_c - point_b
        dot = np.dot(vector_ba, vector_bc)
        magnitude_ab = np.linalg.norm(vector_ba)
        magnitude_bc = np.linalg.norm(vector_bc)
        return np.degrees(np.arccos(dot / (magnitude_ab * magnitude_bc + 1e-6)))

    def kinematic_joint(self):
        #base on 3 division axis provide the degree of angle
        #select the important points to be the angle
        joint_angles = {}
        # shoulder angle (ear-shoulder-elbow)
        joint_angles["left_shoulder"] = self.calculate_angles(
            self.human_left_ear, self.human_left_shoulder, self.human_left_elbow
        )
        joint_angles["right_shoulder"] = self.calculate_angles(
            self.human_right_ear, self.human_right_shoulder, self.human_right_elbow
        )

        # elbow angle (shoulder-elbow-wrist)
        joint_angles["left_elbow"] = self.calculate_angles(
            self.human_left_shoulder, self.human_left_elbow, self.human_left_wrist
        )
        joint_angles["right_elbow"] = self.calculate_angles(
            self.human_right_shoulder, self.human_right_elbow, self.human_right_wrist
        )

        # hip angle (shoulder-hip-knee)
        joint_angles["left_hip"] = self.calculate_angles(
            self.human_left_shoulder, self.human_left_hip, self.human_left_knee
        )
        joint_angles["right_hip"] = self.calculate_angles(
            self.human_right_shoulder, self.human_right_hip, self.human_right_knee
        )

        # knee angle (hip-knee-ankle)
        joint_angles["left_knee"] = self.calculate_angles(
            self.human_left_hip, self.human_left_knee, self.human_left_ankle
        )
        joint_angles["right_knee"] = self.calculate_angles(
            self.human_right_hip, self.human_right_knee, self.human_right_ankle
        )

        # ankle angle (knee-ankle-heel）
        joint_angles["left_ankle"] = self.calculate_angles(
            self.human_left_knee, self.human_left_ankle, self.human_left_heel
        )
        joint_angles["right_ankle"] = self.calculate_angles(
            self.human_right_knee, self.human_right_ankle, self.human_right_heel
        )

        return joint_angles

    def create_angle_mapping(self, human_angles):
        for human_joint, angle in human_angles.items():
            robot_joint = f"robot_{human_joint}_angle"
            self.angle_mapping[robot_joint] = angle

    def normalize_keypoint(self):
        # make x and y key point between 1 and 0
        # eliminate disturb of no related value
        # base human : calculate the distance between two shoulder
        left_shoulder = getattr(self, "human_left_shoulder")
        right_shoulder = getattr(self, "human_right_shoulder")

        # get the middel point and width of shoulder
        middle_x = (left_shoulder.x + right_shoulder.x) / 2
        middle_y = (left_shoulder.y + right_shoulder.y) / 2
        middle_z = (left_shoulder.z + right_shoulder.z) / 2

        shoulder_width = np.linalg.norm([
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
            right_shoulder.z - left_shoulder.z])
        shoulder_width = max(shoulder_width, 1e-6)

        for landmark_enum in mp_pose.PoseLandmark:
            human_joint = f"human_{landmark_enum.name.lower()}"
            joint = getattr(self, human_joint)

            # calculate the gap and normalize data
            norm_x = (joint.x - middle_x) / shoulder_width
            norm_y = (joint.y - middle_y) / shoulder_width
            norm_z = (joint.z - middle_z) / shoulder_width

            setattr(self, f"normalized_{human_joint}", {
                "x": norm_x,
                "y": norm_y,
                "z": norm_z,
            })


def extract_video_data(video_path, save_dir="basic_data", sample_id="001"):
    # extract key points  and ankle data from video
    os.makedirs(save_dir, exist_ok=True)
    motion = MotionDetector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kp = {}
    all_angles = {}

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                # get the key points and angle of this fps
                kp, angles = motion.extract_frame_data(results.pose_landmarks.landmark)
                all_kp.append(kp)
                all_angles.append(angles)
            else:
                # if not detect put zero in it
                all_kp.append(np.zeros((33, 3)))
                all_angles.append(np.zeros(10))

    cap.release()
    # transform to numpy array
    all_kp = np.array(all_kp)
    all_angles = np.array(all_angles)

    # save data
    save_path = os.path.join(save_dir, f"dance_{sample_id}.npz")
    np.savez(save_path,
             keypoints=all_kp,
             joint_angles=all_angles,
             fps=fps,
             total_frames=len(all_kp))
    print(f"basic data has been saved: {save_path}")
    return save_path

if __name__=='__main__':
    motion = MotionDetector()

    # get the video
    input_video = "dance_video.mp4"
    cap = cv2.VideoCapture(input_video)
    print("Press 'q' to exit.")
    print("Press 's' to pause.")

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

            black_image = np.zeros_like(frame)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    black_image, results, mp_pose.POSE_CONNECTIONS
                )

            motion.detect_all_keypoints(results.pose_landmarks.landmark)
            motion.normalize_keypoint()
            human_angles = motion.kinematic_joint()
            robot_angles = motion.create_angle_mapping(human_angles)

            scale = 0.8
            frame = cv2.resize(frame, (0, 0), fx = scale, fy = scale)
            cv2.imshow("robot_dance", frame)
            black_image = cv2.resize(black_image, (0, 0), fx = scale, fy = scale)
            cv2.imshow("black_image", black_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cv2.waitKey(0)
                print("Video has been paused.")

        cap.release()
        cv2.destroyAllWindows()