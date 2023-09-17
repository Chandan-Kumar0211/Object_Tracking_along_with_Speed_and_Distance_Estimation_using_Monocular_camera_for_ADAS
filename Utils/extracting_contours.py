import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 obj_dect
obj_seg = YOLO('yolov8m-seg.pt')

img_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources\demo" \
           r".png"

result_mask = obj_seg(img_path)
# print(result_mask)

masks = result_mask[0].masks
print(len(masks.xy))
org_img = result_mask[0].orig_img
print(org_img.shape)
gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

bboxes = result_mask[0].boxes.xyxy.cpu().numpy().astype(int)
print(len(bboxes))

contour = masks.xy[2]
# print(contours)
contour = contour.astype(np.int32)

# Visualize the results on the frame using plot function provided by ultralytics
annotated_frame = result_mask[0].plot(conf=False)

# Create a blank mask with zeros of the same size as the original image
mask = np.zeros(org_img.shape[:2], dtype=np.uint8)

# Set the segmented region pixels to white (255) in the mask
mask[contour[:, 1], contour[:, 0]] = 255
# Extract the segmented regions from the original image using the mask
contour_on_mask = cv2.bitwise_and(gray_img, mask)

# ======================= Alternative way to draw contours ==================== #
# contour = masks.xy[2]
# contour = [np.array(contour, dtype=np.int32)]
# contour_on_mask = cv2.drawContours(mask, contour, -1, (255,255,255), thickness=2)

# Display the extracted segmented regions
cv2.imshow('Segmented Regions', contour_on_mask)
# Display the  frame
cv2.imshow("Masks", annotated_frame)
cv2.waitKey(0)

cv2.destroyAllWindows()



# ============================= Output of: print(result_mask) ===================================== #
"""
[ultralytics.yolo.engine.results.Results object with attributes:

boxes: ultralytics.yolo.engine.results.Boxes object 

keypoints: None 

keys: ['boxes', 'masks'] 

masks: ultralytics.yolo.engine.results.Masks object 

names: 
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 
28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


--------------------------------------------------------------------------------------------------------------------

Args:
        orig_img (numpy.ndarray): The original image as a numpy array.
        path (str): The path to the image file.
        names (dict): A dictionary of class names.
        boxes (List[List[float]], optional): A list of bounding box coordinates for each detection.
        masks (numpy.ndarray, optional): A 3D numpy array of detection masks, where each mask is a binary image.
        probs (numpy.ndarray, optional): A 2D numpy array of detection probabilities for each class.
        keypoints (List[List[float]], optional): A list of detected keypoints for each object.


    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (tuple): The original image shape in (height, width) format.
        boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
        masks (Masks, optional): A Masks object containing the detection masks.
        probs (numpy.ndarray, optional): A 2D numpy array of detection probabilities for each class.
        names (dict): A dictionary of class names.
        path (str): The path to the image file.
        keypoints (List[List[float]], optional): A list of detected keypoints for each object.
        speed (dict): A dictionary of preprocess, inference and postprocess speeds in milliseconds per image.
        _keys (tuple): A tuple of attribute names for non-empty attributes.
"""