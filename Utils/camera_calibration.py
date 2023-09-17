import cv2
import numpy as np
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (8, 6)
frameSize = (1240, 720)

# Defining termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Creating vector to store vectors of 3D points for each checkerboard image
obj_points = []
# Creating vector to store vectors of 2D points for each checkerboard image
img_points = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
"""
This will output our 3D points like:
    array([[[0., 0., 0.],
        [1., 0., 0.],
        [2., 0., 0.],
        [3., 0., 0.],
        .
        .
        .
        [6., 2., 0.],
        [7., 2., 0.],
        [0., 3., 0.],
        [1., 3., 0.],
        .
        .
        .
        [4., 5., 0.],
        [5., 5., 0.],
        [6., 5., 0.],
        [7., 5., 0.]]], dtype=float32)
"""

prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('../Resources/camera_calibration/*.JPG')
for i, file_name in enumerate(images):
    img = cv2.imread(file_name)
    # print("I'm in: ", i + 1)
    # print(img.shape, "\n")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret:
        obj_points.append(objp)
        # refining pixel coordinates for given 2d points.
        refined_corner = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        img_points.append(refined_corner)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, refined_corner, ret)

    img = cv2.resize(img, frameSize)
    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

"""
Performing camera calibration by 
passing the value of known 3D points (obj_points)
and corresponding pixel coordinates of the 
detected corners (img_points)
"""
ret, mtx, dist, rotation_vectors, translation_vector = cv2.calibrateCamera(obj_points, img_points,
                                                                           gray.shape[::-1], None, None)

# print("Camera matrix : \n")
# print(mtx)
# print("Distortion Parameters : \n")
# print(dist)
# print("rotation_vectors : \n")
# print(rotation_vectors)
# print("translation_vector : \n")
# print(translation_vector)

"""
Camera matrix : 

[[1.10190051e+03 0.00000000e+00 1.00282646e+03]
 [0.00000000e+00 1.10333595e+03 7.62673084e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
 
 
 
Distortion Parameters : 

[[-4.02049257e-01  2.06995701e-01 -3.94682806e-04 -1.01677466e-03
  -5.42424167e-02]]
"""


# ==================== Calculating Reprojection Error ============================== #
total_error = 0

for i in range(len(obj_points)):
    img_point_estimated, _ = cv2.projectPoints(objectPoints=obj_points[i],
                                               rvec=rotation_vectors[i],
                                               tvec=translation_vector[i],
                                               cameraMatrix=mtx,
                                               distCoeffs=dist)
    error = cv2.norm(img_points[i], img_point_estimated, cv2.NORM_L2,) / len(img_point_estimated)
    total_error += error
    print(f"Error of image {i+1} is : {error}")

mean_error = total_error/len(obj_points)
print(f"\nMean Error is : {mean_error}")

