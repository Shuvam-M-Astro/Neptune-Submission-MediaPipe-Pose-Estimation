import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# --------------------------------------
# Feature Extraction from Position Data
# --------------------------------------
def extract_features(positions):
    """
    Extract motion features from a list of (x, y) positions.
    Returns amplitude, velocity, and acceleration stats.
    """
    velocity = np.diff(positions, axis=0)
    acceleration = np.diff(velocity, axis=0)

    amplitude = np.ptp(positions, axis=0).mean()
    mean_velocity = np.mean(np.linalg.norm(velocity, axis=1))
    std_velocity = np.std(np.linalg.norm(velocity, axis=1))
    mean_acceleration = np.mean(np.linalg.norm(acceleration, axis=1))
    std_acceleration = np.std(np.linalg.norm(acceleration, axis=1))

    return [amplitude, mean_velocity, std_velocity, mean_acceleration, std_acceleration]

# --------------------------------------
# Extract right wrist positions from video
# --------------------------------------
def extract_wrist_positions(video_path):
    cap = cv2.VideoCapture(video_path)
    wrist_positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            wrist_positions.append((right_wrist.x, right_wrist.y))

    cap.release()
    return np.array(wrist_positions)

# --------------------------------------
# Add noise to steady positions to simulate shakiness
# --------------------------------------
def simulate_shaky_positions(positions, intensity=0.005):
    noise = np.random.normal(0, intensity, positions.shape)
    return positions + noise

# --------------------------------------
# Plot wrist trajectory and save to file
# --------------------------------------
def plot_wrist_movement(positions, title, filename):
    positions = np.array(positions)
    plt.figure(figsize=(8, 4))
    plt.plot(positions[:, 0], label='X')
    plt.plot(positions[:, 1], label='Y')
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Normalized Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --------------------------------------
# Training Phase with 10-frame segments
# --------------------------------------
videos = ['Violin A.mp4', 'Violin B.mp4']
X_train, y_train = [], []
window_size = 10

for video in videos:
    steady_positions = extract_wrist_positions(video)
    shaky_positions = simulate_shaky_positions(steady_positions)

    plot_wrist_movement(steady_positions, f"Steady Wrist Movement - {video}", f"plot_steady_{video.replace('.mp4', '')}.png")
    plot_wrist_movement(shaky_positions, f"Shaky Wrist Movement - {video}", f"plot_shaky_{video.replace('.mp4', '')}.png")

    for i in range(0, len(steady_positions) - window_size, window_size):
        steady_segment = steady_positions[i:i+window_size]
        shaky_segment = shaky_positions[i:i+window_size]

        X_train.append(extract_features(steady_segment))
        y_train.append(0)

        X_train.append(extract_features(shaky_segment))
        y_train.append(1)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --------------------------------------
# Prediction and Annotated Video Output
# --------------------------------------
input_video = 'Violin C.mp4'
output_video = 'Violin_C_Annotated.mp4'
cap = cv2.VideoCapture(input_video)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

positions_test = extract_wrist_positions(input_video)
predictions = []

# Predict per 10-frame segment
for i in range(0, len(positions_test) - window_size, window_size):
    segment = positions_test[i:i+window_size]
    features_segment = extract_features(segment)
    prediction = clf.predict([features_segment])[0]
    predictions.extend(['Shaky' if prediction else 'Steady'] * window_size)

# Annotate video frames with prediction
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(predictions):
        break

    state_text = predictions[frame_idx]
    color = (0, 0, 255) if state_text == 'Shaky' else (0, 255, 0)
    cv2.putText(frame, f"{state_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Annotated video saved as {output_video}")
