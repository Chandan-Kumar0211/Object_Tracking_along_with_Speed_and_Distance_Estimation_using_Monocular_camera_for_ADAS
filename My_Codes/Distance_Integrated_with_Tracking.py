import torch
import cv2
import numpy as np
from ultralytics import YOLO
from Midas_depth_estimator import MidasDepthEstimator

# Initializing our depth_estimator
# dep_est = MidasDepthEstimator(model_type="MiDaS_small")
# dep_est = MidasDepthEstimator(model_type="DPT_Hybrid")
dep_est = MidasDepthEstimator(model_type="DPT_Large")

# Initializing our Object detection and segmentation model
model = YOLO('yolov8m-seg.pt')


# Converting Depth to distance
def depth_to_distance(inverse_rel_dist, depth_scale=0.00328, shift=-0.05350):
    inverse_absolute_dist = inverse_rel_dist * depth_scale + shift
    dist_absolute = 1 / inverse_absolute_dist
    return round(dist_absolute, 2)


def mean_10_trimmed(arr):
    # flattening the array
    flat_arr = arr.flatten()
    # # removing the NaN values (if any) and zero values (if any)
    # flat_arr = flat_arr[np.logical_not(np.isnan(flat_arr))]
    flat_arr = flat_arr[flat_arr != 0]
    # sorting the array and trimming the top 5 and bottom 5 values
    sort_arr = np.sort(flat_arr)
    trimmed_10_array = sort_arr[5:-5]
    return np.mean(trimmed_10_array)


# Open the video file
video_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
             r"\test_videos\city.mp4"
video = cv2.VideoCapture(video_path)

# Loop through the video frames
while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()
    # print(frame.shape)
    frame = cv2.resize(frame, (720, 540))

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results_track = model.track(frame, persist=True)
        """
        If we want to use the tracker with a folder of images or when we loop on the video frames, we should use 
        the persist parameter to tell the obj_dect that these frames are related to each other so the IDs will be fixed 
        for the same objects. Otherwise, the IDs will be different in each frame because in each loop, the obj_dect 
        creates a new object for tracking, but the persist parameter makes it use the same object for tracking.
        """

        if results_track[0].boxes.id is None:
            continue
        else:
            bboxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
            tracking_ids = results_track[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results_track[0].boxes.cls.cpu().numpy().astype(int)

        class_dict = {0: "person",
                      1: "bicycle",
                      2: "car",
                      3: "motorcycle",
                      5: "bus",
                      6: "train",
                      7: "truck"}

        # processing the frames of video using Midas depth estimator to get the depth image
        depth_img_gray, depth_img_colored = dep_est.process_frame(frame)

        # performing instance segmentation
        results_masks = model(frame)
        # extracting masks region from it
        contours = results_masks[0].masks.xy

        # ============= Finding contours and segments ============ #
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create an empty mask with the same shape as the grayscale image
        empty_mask = np.zeros_like(gray_img)

        if len(bboxes) != len(contours):
            print(f"Length of bboxes: {len(bboxes)}")
            print(f"Length of tracking_ids: {len(tracking_ids)}")
            print(f"Length of class_ids: {len(class_ids)}")
            print(f"Length of contours: {len(contours)}\n")

        final_looping_length = min(len(bboxes), len(contours))

        for i in range(final_looping_length):
            bbox = bboxes[i]
            contour = contours[i]
            cls_id = class_ids[i]
            track_id = tracking_ids[i]

            # Drawing bounding boxes
            if cls_id in class_dict.keys():
                # Separating top-left and bottom-right coordinates
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

                # points for plotting distance on center of bbox
                point_x = x1 + ((bbox[2] - bbox[0]) / 2) - ((bbox[2] - bbox[0]) / 4)
                point_y = y1 + ((bbox[3] - bbox[1]) / 2)

                # The contours variable should be a list of numpy arrays.
                contour = [np.array(contour, dtype=np.int32)]

                # Drawing the contours on the empty_mask
                segmented_mask = cv2.drawContours(empty_mask, contour, -1, 255, thickness=cv2.FILLED)

                # Use the mask to extract the pixels within the contours for all 3 channels of depth map
                extracted_region_blue = cv2.bitwise_and(depth_img_colored[:, :, 0], segmented_mask)
                extracted_region_green = cv2.bitwise_and(depth_img_colored[:, :, 1], segmented_mask)
                extracted_region_red = cv2.bitwise_and(depth_img_colored[:, :, 2], segmented_mask)

                # Use the mask to extract the pixels within the contours for gray depth map
                if depth_img_gray.shape != segmented_mask.shape:
                    print(f"{depth_img_gray.shape}, {segmented_mask.shape}")
                    break
                else:
                    extracted_region_gray = cv2.bitwise_and(depth_img_gray, segmented_mask)

                # Drawing the contours on the both depth_map
                depth_img_gray = cv2.drawContours(depth_img_gray, contour, -1, (255, 255, 255),
                                                  thickness=2, lineType=1)
                depth_img_colored = cv2.drawContours(depth_img_colored, contour, -1, (255, 255, 255),
                                                     thickness=2, lineType=1)

                # calculating mean using our predefined function
                m1 = mean_10_trimmed(extracted_region_blue)
                m2 = mean_10_trimmed(extracted_region_green)
                m3 = mean_10_trimmed(extracted_region_red)

                # total average
                relative_inverse_depth_coloured = (m1 + m2 + m3) / 3
                relative_inverse_depth_gray = mean_10_trimmed(extracted_region_gray)

                # absolute distance
                abs_dist_gray = depth_to_distance(relative_inverse_depth_gray)
                print("Distance from Gray Depth Image", abs_dist_gray)
                abs_dist_colored = depth_to_distance(relative_inverse_depth_coloured)
                print("Distance from Colored Depth Image", abs_dist_colored,"\n")

                # Drawing rectangles and putting required text on our depth images
                # 01: For depth image gray
                cv2.rectangle(depth_img_gray, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(depth_img_gray, f"#{str(track_id)}, {class_dict[cls_id]}",
                            (int(bbox[0]), int(bbox[1] - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(depth_img_gray, str(abs_dist_gray), (int(point_x), int(point_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 01: For depth image coloured
                cv2.rectangle(depth_img_colored, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(depth_img_colored, f"#{str(track_id)}, {class_dict[cls_id]}",
                            (int(bbox[0]), int(bbox[1] - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(depth_img_colored, str(abs_dist_colored), (int(point_x), int(point_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # cv2.imshow("Depth", depth_img_colored)
                # cv2.waitKey(5000)

        cv2.imshow("Depth Image Colored", depth_img_colored)
        cv2.imshow("Depth Image Gray", depth_img_gray)
        cv2.imshow("Original Image", frame)
        cv2.imshow("Segmented Masked Image", segmented_mask)
        # cv2.waitKey(5000)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
