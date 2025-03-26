import cv2
import mediapipe as mp

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open webcam
cap = cv2.VideoCapture(2)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Get right shoulder and elbow
            right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]

            h, w, _ = image.shape
            shoulder_y = int(right_shoulder.y * h)
            elbow_y = int(right_elbow.y * h)
            elbow_x = int(right_elbow.x * w)

            # If elbow above shoulder
            if elbow_y < shoulder_y:
                cv2.circle(image, (elbow_x, elbow_y), 15, (0, 0, 255), -1)  # Red circle
            else:
                cv2.circle(image, (elbow_x, elbow_y), 15, (0, 255, 0), -1)  # Green circle

        # Show the image
        cv2.imshow('Posture Feedback', image)

        # Break on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
