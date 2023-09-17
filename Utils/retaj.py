from ultralytics import YOLO
import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv8 model
# Device is determined automatically. If a GPU is available then it will be used, otherwise training will start on CPU.
model = YOLO('yolov8m.pt')

# # Load the trained classification model
# model_classify = torch.hub.load('pytorch/vision:v0.11.1', 'resnet50', pretrained=False)
# num_classes = 2
# num_features = model_classify.fc.in_features
# model_classify.fc = nn.Linear(num_features, num_classes)
# model_classify.load_state_dict(torch.load('MCClassifyModel.pt'))
# model_classify.eval()

video_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
             r"\test_videos\street.mp4"
video = cv.VideoCapture(video_path)

# Open the video file and Get the video properties
fps = video.get(cv.CAP_PROP_FPS)
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

# Define the output video file path
output_path = 'path_to_output_video_file.mp4'

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Change codec if needed
output_video = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
input_size = (820, 640)

# Loop through the video frames
while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()

    # Resize the frame to the desired input size
    frame = cv.resize(frame, input_size)

    # Convert the frame to RGB format
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # defining width and height
    h, w, _ = frame.shape

    # Convert the frame to a PyTorch tensor
    # frame = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
    if success:
        # Initialize variables for nearest object tracking
        nearest_distance = float('inf')
        nearest_box = None
        nearest_center = None
        nearest_trackID = None

        # Run YOLOv8 detection on the frame
        results_track = model.track(frame, persist=True, )

        if results_track[0].boxes.id is None:
            continue
        else:
            bboxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
            tracking_ids = results_track[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results_track[0].boxes.cls.cpu().numpy().astype(int)

            centre_coord = (int(w / 2), int(h / 2))
            cv.circle(frame, centre_coord, 5, (0, 0, 255), cv.FILLED)
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                cls_id = class_ids[i]
                track_id = tracking_ids[i]
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

                # Calculate the center coordinates
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv.circle(frame, (cx, cy), 3, (20, 255, 160), cv.FILLED)
                cv.line(frame, centre_coord, (cx, cy), (0, 125, 255), 1)

                # Calculate the distance from the center of the frame
                distance = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5
                print(f'Distance of id no #{track_id} from centre is {distance}')

                # Check if the current object is closer to the center
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_box = bbox
                    nearest_center = (cx, cy)
                    nearest_trackID = track_id

            print("\n", nearest_distance, nearest_trackID, "\n")
            if nearest_box is not None:
                x1, y1, x2, y2 = nearest_box

                # Normalize the coordinates
                norm_cx = (nearest_center[0] - frame_width / 2) / frame_width
                norm_cy = (nearest_center[1] - frame_height / 2) / frame_height

                norm_cx = norm_cx - 0.5
                norm_cy = norm_cy - 0.5

                # Print the normalized position of the nearest object
                print(f"(X,Y) ({norm_cx:.2f}, {norm_cy:.2f})")

                # Draw the bounding box, class label, and coordinates on the frame
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(frame, f" X {norm_cx:.2f}, Y {norm_cy:.2f}",
                           (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame to the output video file
            output_video.write(frame)

            # Display the frame
            cv.imshow('Frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        break

# Release the video capture and writer
video.release()
output_video.release()

# Destroy all OpenCV windows
cv.destroyAllWindows()
