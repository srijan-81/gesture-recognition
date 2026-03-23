# Hand Gesture Recognition

Real time finger counting using MediaPipe & OpenCV

## How it works
Uses MediaPipe's hand landmark detection to identify 21 pointson the hand.
Finger state is determined by comparing fingertip and knuckle landmark positions.

## Requirements
- Python 3.10
- mediapipe==0.10.13
- opencv-python

## Usage
python gesture.py
