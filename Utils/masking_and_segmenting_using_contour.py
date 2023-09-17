import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 obj_dect
obj_seg = YOLO('yolov8m-seg.pt')

img_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources\demo" \
           r".png"

result_mask = obj_seg(img_path)
masks = result_mask[0].masks

org_img = result_mask[0].orig_img
print(org_img.shape)
gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image, (720, 540))

# The contours variable should be a list of numpy arrays.
contours = [np.array(masks.xy[2],dtype=np.int32)]
# print(contours)

# Create an empty mask with the same shape as the grayscale image
empty_mask = np.zeros_like(gray_image)

# Draw the contours on the empty_mask
cv2.drawContours(empty_mask, contours, -1, 255, thickness=cv2.FILLED)

# Use the empty_mask to extract the pixels within the contours
extracted_region = cv2.bitwise_and(gray_image, empty_mask)

# Visualize the results on the frame using plot function provided by ultralytics
annotated_frame = result_mask[0].plot(conf=False)
# Display the extracted segmented regions
cv2.imshow('Masked', empty_mask)
cv2.imshow('Extracted Regions', extracted_region)
cv2.imshow("Model Output", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()