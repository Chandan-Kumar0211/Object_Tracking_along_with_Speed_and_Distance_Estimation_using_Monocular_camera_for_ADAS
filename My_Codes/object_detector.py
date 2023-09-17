import torch
import numpy as np
from ultralytics import YOLO

class YoloDetector:

    def __init__(self):
        self.obj_dect_model = self.load_model()
        self.classes = self.obj_dect_model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\nUsing Device:", self.device)

    # loading the model
    def load_model(self):
        model = YOLO("yolov8s.pt")
        return model

    # Getting the top left and bottom right corner of the bounding box
    def get_tl_br_cords_and_labels(self, frame):
        self.obj_dect_model = self.obj_dect_model.to(self.device)
        frame = [frame]
        results = self.obj_dect_model(frame)

        full_cord = results.xyxy[0]     # [x1, y1, x2, y2, confidence, class_label]
        class_labels = full_cord[:, -1]
        tl_br = full_cord[:, :4]
        return class_labels, tl_br

    # Performing Forward pass on each frame
    def get_center_cords_and_labels(self, frame):
        self.obj_dect_model = self.obj_dect_model.to(self.device)
        frame = [frame]
        results = self.obj_dect_model(frame)

        labels = results.xyxyn[0][:, -1]
        # tensor([ 2.,  0.,  1.,  0.,  9.,  1.,  1.,  9., 26.,  9.,  9., 26.,  1.], device='cuda:0')
        cord = results.xyxyn[0][:, :-1]
        # tensor([[0.24218, 0.44742, 0.36322, 0.76161, 0.88225],
        #         [0.89210, 0.46580, 0.96698, 0.84235, 0.87286],
        #         [0.61950, 0.67974, 0.71250, 0.99007, 0.85827],
        #         [0.62714, 0.44854, 0.70221, 0.95510, 0.83061],
        #         [0.10083, 0.05403, 0.13362, 0.34677, 0.70514],
        #         [0.59984, 0.49819, 0.63526, 0.62431, 0.68666],
        #         [0.71582, 0.52177, 0.77154, 0.63519, 0.64936],
        #         [0.75082, 0.31701, 0.76565, 0.38125, 0.51751],
        #         [0.68275, 0.58473, 0.71125, 0.75530, 0.51660],
        #         [0.01496, 0.24461, 0.06505, 0.34093, 0.44977],
        #         [0.77826, 0.26261, 0.80152, 0.39014, 0.42836],
        #         [0.93753, 0.63764, 0.95778, 0.71515, 0.37800],
        #         [0.44553, 0.52338, 0.47821, 0.60138, 0.37306]], device='cuda:0')
        return labels, cord


    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_bboxes(self, label, coord, frame, frame_height, frame_width, confidence=0.3):
        detections = []
        x_shape, y_shape = frame_width, frame_height

        for i in range(len(label)):
            row = coord[i]
            # print(row)

            if row[4] > confidence:
                x1, y1 = int(row[0] * x_shape), int(row[1] * y_shape)
                x2, y2 = int(row[2] * x_shape), int(row[3] * y_shape)

                #   0: person
                #   1: bicycle
                #   2: car
                #   3: motorcycle
                #   4: airplane
                #   5: bus
                #   6: train
                #   7: truck

                focus_objects = ["person", "car"]
                if self.class_to_label(label[i]) == "person":
                    # x_center = x1 + ((x2 - x1) / 2)
                    # y_center = y1 + ((y2 - y1) / 2)

                    tl_wh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = 'person'

                    detections.append((tl_wh, confidence, feature))

                elif self.class_to_label(label[i]) == "car":
                    # x_center = x1 + ((x2 - x1) / 2)
                    # y_center = y1 + ((y2 - y1) / 2)

                    tl_wh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = 'car'

                    detections.append((tl_wh, confidence, feature))
                else:
                    continue

        return frame, detections

