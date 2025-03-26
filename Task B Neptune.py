import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between 3D vectors
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # prevent numerical errors
    return np.degrees(angle)

# Define reference posture angles (in degrees)
# These should be determined from a "good" violin posture sample
reference_posture = {
    "right_arm": 90,  # shoulder-elbow-wrist
    "left_arm": 120
}

tolerance = 15  # degrees

# Open webcam
cap = cv2.VideoCapture(2)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            def get_coords(part):
                lm = landmarks[mp_pose.PoseLandmark[part]]
                return [lm.x, lm.y, lm.z]

            # Extract joints
            rs = get_coords("RIGHT_SHOULDER")
            re = get_coords("RIGHT_ELBOW")
            rw = get_coords("RIGHT_WRIST")

            ls = get_coords("LEFT_SHOULDER")
            le = get_coords("LEFT_ELBOW")
            lw = get_coords("LEFT_WRIST")

            # Calculate angles
            right_arm_angle = calculate_angle(rs, re, rw)
            left_arm_angle = calculate_angle(ls, le, lw)

            # Compare with reference
            right_flag = abs(right_arm_angle - reference_posture["right_arm"]) > tolerance
            left_flag = abs(left_arm_angle - reference_posture["left_arm"]) > tolerance

            # Visual feedback
            color_r = (0, 0, 255) if right_flag else (0, 255, 0)
            color_l = (0, 0, 255) if left_flag else (0, 255, 0)

            h, w, _ = image.shape
            re_x, re_y = int(re[0]*w), int(re[1]*h)
            le_x, le_y = int(le[0]*w), int(le[1]*h)

            cv2.putText(image, f'R-angle: {int(right_arm_angle)}',
                        (re_x - 50, re_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_r, 2)
            cv2.circle(image, (re_x, re_y), 10, color_r, -1)

            cv2.putText(image, f'L-angle: {int(left_arm_angle)}',
                        (le_x - 50, le_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_l, 2)
            cv2.circle(image, (le_x, le_y), 10, color_l, -1)

        cv2.imshow('Violin Posture Feedback', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
