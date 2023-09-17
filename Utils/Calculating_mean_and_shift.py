import cv2
import numpy as np
from ultralytics import YOLO
from My_Codes.Midas_depth_estimator import MidasDepthEstimator


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


# Initializing our depth_estimator
# dep_est = MidasDepthEstimator(model_type="MiDaS_small")
# dep_est = MidasDepthEstimator(model_type="DPT_Hybrid")
dep_est = MidasDepthEstimator(model_type="DPT_Large")


# Initializing our segmentation model
obj_seg = YOLO('yolov8m-seg.pt')

img_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
           r"\scale_shift\Screenshot_42.png"
frame = cv2.imread(img_path)
frame = cv2.resize(frame, (720,540))

result_mask = obj_seg(frame)
masks = result_mask[0].masks

gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# processing the frame using Midas depth estimator to get the depth image
depth_img_gray, depth_img_coloured = dep_est.process_frame(frame)


# The contours variable should be a list of numpy arrays.
# Here, choose k such that masks.xy[k] is a person class as we know only persons class absolute distance
contours = [np.array(masks.xy[2], dtype=np.int32)]
# print(masks.xy)

# Create an empty mask with the same shape as the grayscale image
empty_mask = np.zeros_like(gray_image)

# Draw the contours on the empty_mask
segmented_mask = cv2.drawContours(empty_mask, contours, -1, 255, thickness=cv2.FILLED)
print(segmented_mask.shape)

# Use the segmented_mask to extract the pixels within the contours from gray scale image
gray_extracted_region = cv2.bitwise_and(gray_image, segmented_mask)

# Use the segmented_mask to extract the pixels within the contours from depth map
depth_extracted_region = cv2.bitwise_and(depth_img_gray, segmented_mask)

# calculating mean using our predefined function
relative_inverse_depth = mean_10_trimmed(depth_extracted_region)
print("Relative Inverse Depth", relative_inverse_depth)

# Visualize the results on the frame using plot function provided by ultralytics
annotated_frame = result_mask[0].plot(conf=False)

# Display the extracted segmented regions
cv2.imshow('Masked', segmented_mask)
cv2.imshow('Extracted Regions', depth_extracted_region)
cv2.imshow("Model Output", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()



"""


"""