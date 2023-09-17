import cv2
from ultralytics import YOLO

# Load the YOLOv8 obj_dect
obj_dect = YOLO('yolov8s.pt')
# obj_dect = YOLO('yolov8m-seg.pt')

# Open the video file
video_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
             r"\street.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    frame = cv2.resize(frame, (924, 640))

    if success:

        predictions = obj_dect(frame)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = obj_dect.track(frame, persist=True)
        """
        If we want to use the tracker with a folder of images or when we loop on the video frames, we should use 
        the persist parameter to tell the obj_dect that these frames are related to each other so the IDs will be fixed 
        for the same objects. Otherwise, the IDs will be different in each frame because in each loop, the obj_dect 
        creates a new object for tracking, but the persist parameter makes it use the same object for tracking.

        """
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if results[0].boxes.id is None:
            continue
        else:
            tracking_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            #   0: person
            #   1: bicycle
            #   2: car
            #   3: motorcycle
            #   4: airplane
            #   5: bus
            #   6: train
            #   7: truck

            class_dict = {0: "person",
                          1: "bicycle",
                          2: "car",
                          3: "motorcycle",
                          5: "bus",
                          6: "train",
                          7: "truck"}

        # for box, track_id, cls_id in zip(boxes, tracking_ids, class_ids):
        #     if cls_id in class_dict.keys():
        #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #         cv2.putText(frame, f"#{track_id}, {class_dict[cls_id]}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (0, 0, 255), 2)

        # Visualize the results on the frame using plot function provided by ultralytics
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
