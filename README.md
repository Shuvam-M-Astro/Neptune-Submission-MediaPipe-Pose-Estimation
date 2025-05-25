# Real-Time Pose Estimation and Feedback for Violinists

This project focuses on real-time body tracking for violinists using MediaPipe for pose estimation and provides feedback on their posture and movements. It includes multiple tasks related to motion tracking, posture evaluation, and using machine learning for motion classification.

## Tasks

### Task A: Real-Time Posture Feedback - Tracking Bowing Arm Movement

**Overview**

Task A aims to track the right bowing arm's movement and provide real-time feedback based on the elevation of the elbow compared to the shoulder. If the elbow is raised above the shoulder, the posture is flagged, and visual feedback (color change) is provided to the user.

**Features**

* **Live Pose Detection**: Uses MediaPipe to detect human body landmarks in real-time.
* **Landmark Tracking**: Tracks left shoulder and left elbow positions from each frame.
* **Posture Evaluation**: Compares vertical positions of the elbow and shoulder to check if the elbow is raised above the shoulder.
* **Visual Feedback**: If the elbow is raised above the shoulder, a red circle is drawn; otherwise, a green circle is displayed.

#### Demo Video:
[Watch the demo video for Task A](https://drive.google.com/file/d/1-qsEet-vwrD8QSecjFgiGkP6VNFDg-CS/view?usp=drive_link)

**Running the Task**

1.  Run the Task:

    ```bash
    git clone https://github.com/Shuvam-M-Astro/Neptune-Submission-MediaPipe-Pose-Estimation.git
    cd Neptune-Submission-MediaPipe-Pose-Estimation
    pip install mediapipe opencv-python
    python TaskA_Neptune.py
    ```

### Task B: 3D Pose Estimation - Tracking Violin Posture

**Overview**

Task B calculates the joint angles for the upper body (shoulder, elbow, wrist) using 3D pose estimation. It compares these angles to a predefined reference posture for violin playing and provides feedback if there's any deviation from the ideal angles.

**Features**

* **3D Pose Detection**: Uses MediaPipe Pose to detect the 3D coordinates of key body landmarks (shoulders, elbows, wrists).
* **Angle Calculation**: Calculates joint angles for both right and left arms using vector math in 3D space.
* **Reference Comparison**: Compares live arm angles to predefined reference posture angles.
* **Visual Feedback**: Shows angle values in real-time and flags any deviations from the reference angles.

#### Demo Video:
[Watch the demo video for Task B](https://drive.google.com/file/d/1INlvHgj14Wu5y-Trw-PXVSD4Kqs8fJHA/view?usp=drive_link)


**Running the Task**

1.  Run the Task:

    ```bash
    git clone https://github.com/Shuvam-M-Astro/Neptune-Submission-MediaPipe-Pose-Estimation.git
    cd Neptune-Submission-MediaPipe-Pose-Estimation
    pip install mediapipe opencv-python numpy
    python TaskB_Neptune.py
    ```

### Task D: BlazePose-Based Motion Tracker - Shoulder Elevation Detection

**Overview**

This task uses BlazePose via TensorFlow.js to detect shoulder elevation during bowing. The system implements temporal smoothing to reduce jitter and provide more stable feedback.

**Features**

* **BlazePose Pose Detection**: Uses TensorFlow.js to run the BlazePose model directly in the browser.
* **Temporal Smoothing**: Applies an exponential moving average (EMA) to smooth the vertical shoulder positions and reduce jitter.
* **Shoulder Elevation Detection**: Compares the smoothed shoulder positions to determine if one shoulder is elevated during bowing.
* **Console Feedback**: Logs the smoothed shoulder positions and their elevation status.

#### Demo Video:
[Watch the demo video for Task D](https://drive.google.com/file/d/1GMIfspa2m1lc5bdMmxDCNVFd10LDX8uj/view?usp=drive_link)

**Running the Task**

1.  Clone the repository and navigate to the web folder:

    ```bash
    git clone https://github.com/Shuvam-M-Astro/Neptune-Submission-MediaPipe-Pose-Estimation.git
    cd Neptune-Submission-MediaPipe-Pose-Estimation/web
    ```

2.  Install dependencies using npm:

    ```bash
    npm install tensorflow
    ```

3.  Open `index.html` in a browser, and check the console (press F12) to see the real-time shoulder elevation status.

### Task H: Violin Movement Classification - Steady vs Shaky Bowing

**Overview**

This task uses a machine learning model (Random Forest) to classify wrist movements in a violin performance as "steady" or "shaky." The model is trained using wrist movement data extracted from MediaPipe Pose landmarks.

**Features**

* **Motion Feature Extraction**: Extracts wrist movement data (amplitude, velocity, acceleration) from MediaPipe Pose landmarks.
* **Shakiness Classification**: Classifies wrist movement as steady or shaky using a Random Forest classifier.
* **Visualization**: Annotates the video with "steady" or "shaky" labels for each 10-frame segment.

**Running the Task**

1.  Run the Task:

    ```bash
    git clone https://github.com/Shuvam-M-Astro/Neptune-Submission-MediaPipe-Pose-Estimation.git
    cd Neptune-Submission-MediaPipe-Pose-Estimation
    pip install mediapipe opencv-python scikit-learn
    python TaskH_Classifier.py
    ```

## Project Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/Shuvam-M-Astro/Neptune-Submission-MediaPipe-Pose-Estimation.git
    cd Neptune-Submission-MediaPipe-Pose-Estimation

    ```

2.  Install dependencies:
    Use the following to install all necessary Python dependencies for the project:

    ```bash
    pip install -r requirements.txt
    ```
3.  For the web-based task, navigate to the web directory and run the necessary commands:

    ```bash
    npm install tensorflow
    ```

3.  Run the script: Follow the instructions for each task to run the corresponding script.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or new features. Please make sure to follow best practices for code quality and documentation.
