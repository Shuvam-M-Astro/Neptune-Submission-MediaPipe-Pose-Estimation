import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Constants
WEBCAM_INDEX = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
REFERENCE_POSTURE = {
    "right_arm": 90,  # shoulder-elbow-wrist
    "left_arm": 120
}
TOLERANCE = 15  # degrees

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

# Function to extract coordinates of a joint from landmarks
def get_coords(landmarks, part):
    lm = landmarks[mp_pose.PoseLandmark[part]]
    return [lm.x, lm.y, lm.z]

# Function to provide visual feedback on posture
def provide_visual_feedback(image, angle, side, color, label, coords):
    h, w, _ = image.shape
    x, y = int(coords[0] * w), int(coords[1] * h)
    cv2.putText(image, f'{label}-angle: {int(angle)}',
                (x - 50, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.circle(image, (x, y), 10, color, -1)

# Function to start pose detection and feedback
def start_violin_posture_feedback():
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    with mp_pose.Pose(min_detection_confidence=DETECTION_CONFIDENCE,
                      min_tracking_confidence=TRACKING_CONFIDENCE) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Flip frame for a mirror view
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates of joints
                rs = get_coords(landmarks, "RIGHT_SHOULDER")
                re = get_coords(landmarks, "RIGHT_ELBOW")
                rw = get_coords(landmarks, "RIGHT_WRIST")

                ls = get_coords(landmarks, "LEFT_SHOULDER")
                le = get_coords(landmarks, "LEFT_ELBOW")
                lw = get_coords(landmarks, "LEFT_WRIST")

                # Calculate angles
                right_arm_angle = calculate_angle(rs, re, rw)
                left_arm_angle = calculate_angle(ls, le, lw)

                # Compare with reference posture
                right_flag = abs(right_arm_angle - REFERENCE_POSTURE["right_arm"]) > TOLERANCE
                left_flag = abs(left_arm_angle - REFERENCE_POSTURE["left_arm"]) > TOLERANCE

                # Provide visual feedback
                right_color = (0, 0, 255) if right_flag else (0, 255, 0)
                left_color = (0, 0, 255) if left_flag else (0, 255, 0)

                # Display feedback
                provide_visual_feedback(image, right_arm_angle, 'R', right_color, 'R', re)
                provide_visual_feedback(image, left_arm_angle, 'L', left_color, 'L', le)

            # Show the image with feedback
            cv2.imshow('Violin Posture Feedback', image)

            # Break loop on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        start_violin_posture_feedback()
    except Exception as e:
        print(f"An error occurred: {e}")
