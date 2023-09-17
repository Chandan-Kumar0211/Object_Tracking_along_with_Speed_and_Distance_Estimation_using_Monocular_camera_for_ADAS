import torch
import numpy as np


"""
A class for storing and manipulating detection masks.

    Args:
        masks (torch.Tensor): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        masks (torch.Tensor): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Properties:
        xy (list): A list of segments (pixels) which includes x, y segments of each detection.
        xyn (list): A list of segments (normalized) which includes x, y segments of each detection.

    Methods:
        cpu(): Returns a copy of the masks tensor on CPU memory.
        numpy(): Returns a copy of the masks tensor as a numpy array.
        cuda(): Returns a copy of the masks tensor on GPU memory.
        to(): Returns a copy of the masks tensor with the specified device and dtype.
"""
class DetectionMasks:
    def __init__(self, masks, orig_shape):
        self.masks = masks
        self.orig_shape = orig_shape

    def cpu(self):
        return DetectionMasks(self.masks.cpu(), self.orig_shape)

    def numpy(self):
        return self.masks.numpy()

    def cuda(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return DetectionMasks(self.masks.cuda(), self.orig_shape)

    def to(self, device, dtype=None):
        return DetectionMasks(self.masks.to(device, dtype=dtype), self.orig_shape)

    @property
    def xy(self):
        xy_segments = []
        num_masks, height, width = self.masks.shape
        for mask in self.masks:
            y, x = np.where(mask.cpu().numpy())  # Assuming masks are on the CPU
            xy_segments.append(list(zip(x.tolist(), y.tolist())))
        return xy_segments

    @property
    def xyn(self):
        xyn_segments = []
        num_masks, height, width = self.masks.shape
        for mask in self.masks:
            y, x = np.where(mask.cpu().numpy())  # Assuming masks are on the CPU
            x_norm = x / (width - 1)
            y_norm = y / (height - 1)
            xyn_segments.append(list(zip(x_norm.tolist(), y_norm.tolist())))
        return xyn_segments
