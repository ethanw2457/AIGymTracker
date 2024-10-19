# AIGymTracker
<h1>Mediapipe Pose Detection in Python</h1>

<p>This project demonstrates how to use the <a href="https://mediapipe.dev/">Mediapipe</a> library for pose detection using a live video feed in Python. The project also includes functionality to calculate the angles of joints and a simple curl counter based on the user's pose.</p>

<h2>Features</h2>
<ul>
  <li><strong>Live Video Feed:</strong> Captures video from your webcam using OpenCV.</li>
  <li><strong>Pose Detection:</strong> Detects body landmarks using Mediapipe's Pose module.</li>
  <li><strong>Angle Calculation:</strong> Calculates the angle between specific body joints (shoulder, elbow, wrist).</li>
  <li><strong>Curl Counter:</strong> Counts repetitions of a bicep curl based on the joint angles.</li>
</ul>

<h2>Requirements</h2>
<p>Make sure you have the following libraries installed:</p>
<pre><code>pip install mediapipe opencv-python numpy</code></pre>

<h2>Running the Code in Jupyter</h2>
<ol>
  <li>Clone this repository to your local machine:</li>
  <pre><code>git clone https://github.com/yourusername/mediapipe-pose-detection.git</code></pre>
  
  <li>Navigate to the project directory:</li>
  <pre><code>cd mediapipe-pose-detection</code></pre>
  
  <li>Open the <code>mediapipe_pose_detection.ipynb</code> notebook in Jupyter:</li>
  <pre><code>jupyter notebook mediapipe_pose_detection.ipynb</code></pre>
  
  <li>Run the notebook cells in sequence. The live video feed will open using your webcam, and Mediapipe will detect your pose in real time.</li>
</ol>

<h2>How It Works</h2>
<ol>
  <li><strong>Pose Detection:</strong> The code uses Mediapipe's <code>mp_pose.Pose()</code> class to process frames from the video feed and detect body landmarks.</li>
  <li><strong>Angle Calculation:</strong> The <code>calculate_angle</code> function calculates the angle between three points (joint positions) to measure the movement of the body.</li>
  <li><strong>Curl Counter:</strong> The curl counter detects if the user is performing a bicep curl based on the angle between the shoulder, elbow, and wrist, and increments the counter when a full curl is completed.</li>
</ol>

<h2>Code Breakdown</h2>
<ul>
  <li><strong>Import Libraries:</strong>
    <p>We import the necessary libraries such as <code>cv2</code> (OpenCV), <code>mediapipe</code>, and <code>numpy</code>.</p>
    <pre><code>import cv2
import mediapipe as mp
import numpy as np
</code></pre>
  </li>

  <li><strong>Pose Detection Setup:</strong>
    <p>We initialize the pose detection system using <code>mp_pose.Pose()</code>. The minimum detection and tracking confidence are set to 0.5.</p>
    <pre><code>mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
</code></pre>
  </li>

  <li><strong>Live Video Feed:</strong>
    <p>The webcam feed is captured using OpenCV, and the pose is processed in real-time. The results are displayed with landmarks and connections drawn on the image.</p>
    <pre><code>cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
</code></pre>
  </li>

  <li><strong>Curl Counter Logic:</strong>
    <p>The program calculates the angle between the shoulder, elbow, and wrist and increments the curl counter based on the angle changes.</p>
    <pre><code>def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
</code></pre>
  </li>
</ul>

<h2>Reference Image</h2>
<p>The points used for the pose detection and angle calculation are based on Mediapipe's landmark model. You can view the points here:</p>
<p><img src="https://i.imgur.com/3j8BPdc.png" alt="Mediapipe Landmarks"></p>

<h2>License</h2>
<p>This project is licensed under the MIT License.</p>




## 📊Profile Wide Statistics📊

![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=ethanw2457&show_icons=true)
![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=cashyup47&show_icons=true)

## 😂Random Joke for visiting this ReadMe!😂
![Jokes Card](https://readme-jokes.vercel.app/api)
 
