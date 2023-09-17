import cv2
import torch
import numpy
import os

from ultralytics import YOLO

obj_seg = YOLO("yolov8s-seg.pt")
# obj_dect = YOLO("yolov8s.pt")


root_directory = r'C:\Users\CHAND\PycharmProjects\Object_Detection_in_PyTorch\00_Resources' \
                 r'\data_tracking_image_2\testing\image_02'

# Iterate over each folder and its sub_folders
for data_tracking_image_2, sub_folders, filenames in os.walk(root_directory):
    # Iterate over each file in the current folder
    for filename in filenames:
        # Check if the file has an image extension
        if filename.endswith('.png'):
            # Get the full path of the image file
            image_path = os.path.join(data_tracking_image_2, filename)

            # Performing operations on the image file here
            frame = cv2.imread(image_path)

            results = obj_seg.track(frame, persist=True)
            # results = obj_dect.track(frame, persist=True)
            # boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            # ids = results[0].boxes.id.cpu().numpy().astype(int)
            # for box, id in zip(boxes, ids):
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #     cv2.putText(frame, f"Id {id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
            #                 1, (0, 0, 255), 2)
            # cv2.imshow("frame", frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
