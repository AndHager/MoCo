import cv2
import mediapipe as mp


class PoseDetector:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    configure = True

    knee_to_shoulder_height_left = 0
    knee_to_shoulder_height_right = 0

    def detect_pose(self, rgb_image):
        pose = self.pose.process(rgb_image)
        if self.is_config():
            self.__config(pose)
        return pose

    def draw_pose_landmarks(self, frame, pos):
        self.mp_draw.draw_landmarks(
            frame,
            pos.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def start_config(self):
        self.configure = True

    def end_config(self):
        self.configure = False

    def is_config(self):
        return self.configure

    def __config(self, pose):
        print("pose:")
        print(pose.pose_landmarks)
        exit(0)

        knee_to_shoulder_height_left = 0
        knee_to_shoulder_height_right = 0

        print(self.knee_to_shoulder_height_left)
        print(self.knee_to_shoulder_height_right)

    def is_ducking(self, pos):
        # pos.pose_landmarks[]
        return True

    def is_jumping(self, pos):
        return True


class FaceDetector:
    mp_face = mp.solutions.face_mesh
    face = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    def detect_face(self, rgb_image):
        return self.face.process(rgb_image)

    def draw_face_landmarks(self, frame, faces):
        if faces.multi_face_landmarks:
            for face_landmarks in faces.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())


class HandDetector:
    mpHands = mp.solutions.hands  # detect hand and fingers
    hands = mpHands.Hands()  # complete the initialization configuration of hands
    mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, rgb_image):
        return self.hands.process(rgb_image)

    def draw_hand_landmarks(self, frame, hands):
        if hands.multi_hand_landmarks:
            for hand_landmark in hands.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmark, self.mpHands.HAND_CONNECTIONS)

    def get_landmark_list(self, frame, hands):
        landmark_list = []
        if hands.multi_hand_landmarks:
            for hand_landmark in hands.multi_hand_landmarks:
                for landmark_id, landmark in enumerate(hand_landmark.landmark):  # adding counter and returning it
                    # Get finger joint points
                    h, w, _ = frame.shape
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)
                    landmark_list.append([landmark_id, cx, cy])  # adding to the empty list 'lmList'
        return landmark_list


def main():
    capture_video()


def capture_video():
    video = cv2.VideoCapture(0)

    hand_detector = HandDetector()
    face_detector = FaceDetector()
    pose_detector = PoseDetector()

    while True:
        check, frame = video.read()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = hand_detector.detect_hands(rgb_image)
        hand_detector.draw_hand_landmarks(frame, hands)

        faces = face_detector.detect_face(rgb_image)
        face_detector.draw_face_landmarks(frame, faces)

        pos = pose_detector.detect_pose(rgb_image)
        pose_detector.draw_pose_landmarks(frame, pos)

        if pose_detector.is_config():
            pass

        cv2.imshow("Color Frame", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            pose_detector.start_config()
        if key == ord('e'):
            pose_detector.end_config()
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Welcome to the Motion Controller MoCo :)")
    print("press 'q' to exit")
    print("press 's' to start configuration")
    print("press 'e' to end configuration and start playing")
    main()
