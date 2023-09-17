import torch
import cv2
import numpy as np

# =========================================== NOTE =========================================== #
"""
The prediction is relative inverse depth. For each prediction, there exist some scalars a,b such 
that a*prediction+b is the absolute inverse depth. 
The factors a,b cannot be determined without additional measurements.
"""
# -------------------------------------------------------------------------------------------- #
# Defining the type of model and Loading the model:
# model_type = "DPT_Large"     # MiDaS v3 - Large     (the highest accuracy, the slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (the lowest accuracy, the highest inference speed)


class MidasDepthEstimator:
    def __init__(self, model_type="DPT_Hybrid"):
        self.model_type = model_type
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type).to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

    def process_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        """
        NOTE: The above depth_map contains values grater than 255 and even after normalizing the values we can't
              directly visualize it as it contains float values so again we need to convert them into unit8.
        """

        # normalizing
        depth_map_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        # float to unit8 and also refactoring values by multiplying with 255
        depth_map_gray = (depth_map_norm * 255).astype(np.uint8)
        # applying colour
        depth_map_colored = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_MAGMA)

        return depth_map_gray, depth_map_colored



# # Example usage
# model = MidasDepthEstimator(model_type="DPT_Large")
#
# img_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources\demo.png"
# frame = cv2.imread(img_path)
#
# _, depth_img = model.process_frame(frame)
#
# print("\n",  depth_img.shape, "\n")
# print(depth_img)
#
# cv2.imshow('original img', frame)
# cv2.imshow('depth img', depth_img)
# cv2.waitKey(0)
