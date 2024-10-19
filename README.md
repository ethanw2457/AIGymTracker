# AIGymTracker
# Mediapipe Pose Detection in Python

This project demonstrates how to use the [Mediapipe](https://mediapipe.dev/) library for pose detection using a live video feed in Python. The project also includes functionality to calculate the angles of joints and a simple curl counter based on the user's pose.

## Features
- **Live Video Feed:** Captures video from your webcam using OpenCV.
- **Pose Detection:** Detects body landmarks using Mediapipe's Pose module.
- **Angle Calculation:** Calculates the angle between specific body joints (shoulder, elbow, wrist).
- **Curl Counter:** Counts repetitions of a bicep curl based on the joint angles.

## Requirements

Make sure you have the following libraries installed:

```bash
pip install mediapipe opencv-python numpy
