import cv2
import math
from ultralytics import YOLO

# Load the YOLOv8 obj_dect
# obj_dect = YOLO('yolov8s.pt')  # small
obj_dect = YOLO('yolov8m-seg.pt')  # medium

speed_line_queue = {}


def estimate_speed(Location1, Location2, img_width, focal_length=1102.61842573):
    # Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining the pixels per meter
    # PPM = (Image Width in Pixels) / (2 * tan(FOV / 2) * Focal Length)
    ppm = img_width/(2*(math.tan(140/2))*focal_length)
    # print("PPM: ",ppm)
    # ppm = 8
    d_meters = d_pixel / ppm
    # time constant = fps * scale; where scale is the constant which we can adjust,
    time_constant = 30 * 0.1
    # speed  = distance/time  = distance * time_constant
    est_speed = d_meters * time_constant
    return round(est_speed,2)


# Defining the objects on focus
class_dict = {0: "person",
              1: "bicycle",
              2: "car",
              3: "motorcycle",
              5: "bus",
              6: "train",
              7: "truck"}


# Defining location of objects in their previous frame
prev_loc = {}

# Open the video file
video_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
             r"\test_videos\street.mp4"
video = cv2.VideoCapture(video_path)

# Loop through the video frames
while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()
    frame = cv2.resize(frame, (924, 640))

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results_track = obj_dect.track(frame, persist=True)
        """
        If we want to use the tracker with a folder of images or when we loop on the video frames, we should use 
        the persist parameter to tell the obj_dect that these frames are related to each other so the IDs will be fixed 
        for the same objects. Otherwise, the IDs will be different in each frame because in each loop, the obj_dect 
        creates a new object for tracking, but the persist parameter makes it use the same object for tracking.
        """
        boxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
        if results_track[0].boxes.id is None:
            continue
        else:
            bboxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
            tracking_ids = results_track[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results_track[0].boxes.cls.cpu().numpy().astype(int)

            # A loop for storing current locations(any one point) of each detected object
            curr_loc = {}
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                cls_id = class_ids[i]
                track_id = tracking_ids[i]

                if cls_id in class_dict.keys():
                    # Separating top-left and bottom-right coordinates
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    # defining the current location
                    curr_loc[track_id] = [x1, y1]

            for i in range(len(bboxes)):
                bbox = bboxes[i]
                cls_id = class_ids[i]
                track_id = tracking_ids[i]

                if cls_id in class_dict.keys():
                    # Separating top-left and bottom-right coordinates
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    x2, y2 = int(bbox[2]), int(bbox[3])

                    # points for plotting distance on center of bbox
                    point_x = x1 + ((bbox[2] - bbox[0]) / 2) - ((bbox[2] - bbox[0]) / 4)
                    point_y = y1 + ((bbox[3] - bbox[1]) / 2)

                    # calculating speed of tracked object which is present in previous frame too
                    if track_id not in prev_loc.keys():
                        speed = 0
                    else:
                        speed = estimate_speed(prev_loc[track_id], curr_loc[track_id], frame.shape[1])

                    # redefining our previous location for speed calculation in next frame
                    prev_loc[track_id] = curr_loc[track_id]

                    # Drawing rectangles and putting required text on our depth images
                    # 01: For depth image gray
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(frame, f"#{str(track_id)}, {class_dict[cls_id]}",
                                (int(bbox[0]), int(bbox[1] - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    cv2.putText(frame, f'{speed} km/h', (int(point_x), int(point_y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Original Image", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()
