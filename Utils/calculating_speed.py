import numpy as np
import cv2
import os
import torch


video_file = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
             r"\test_videos\test_01.mp4"

cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    success, frame = cap.read()
    while success:
        frame = cv2.resize(frame, (540, 880))

        

