import cv2
import mediapipe as mp

# Initialize MediaPipe pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Constants
WEBCAM_INDEX = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

def process_frame(frame, pose_detector):
    """Process a frame for pose detection and provide feedback."""
    # Flip the frame horizontally for a natural selfie view
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose_detector.process(rgb_frame)
    
    # Convert back to BGR for OpenCV
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(bgr_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get right shoulder and elbow
        right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]

        # Get image dimensions
        h, w, _ = bgr_frame.shape
        shoulder_y = int(right_shoulder.y * h)
        elbow_y = int(right_elbow.y * h)
        elbow_x = int(right_elbow.x * w)

        # Determine if the elbow is above the shoulder (bad posture)
        if elbow_y < shoulder_y:
            cv2.circle(bgr_frame, (elbow_x, elbow_y), 15, (0, 0, 255), -1)  # Red circle for bad posture
        else:
            cv2.circle(bgr_frame, (elbow_x, elbow_y), 15, (0, 255, 0), -1)  # Green circle for good posture

    return bgr_frame

def start_pose_detection():
    """Initialize webcam and start pose detection loop."""
    # Open the webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    with mp_pose.Pose(min_detection_confidence=DETECTION_CONFIDENCE,
                      min_tracking_confidence=TRACKING_CONFIDENCE) as pose_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame and get feedback
            processed_frame = process_frame(frame, pose_detector)

            # Display the frame
            cv2.imshow('Posture Feedback', processed_frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        start_pose_detection()
    except Exception as e:
        print(f"An error occurred: {e}")
