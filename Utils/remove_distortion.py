import cv2
import numpy as np

camera_matrix = [[1.10190051e+03, 0.00000000e+00, 1.00282646e+03],
                 [0.00000000e+00, 1.10333595e+03, 7.62673084e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
camera_matrix = np.array(camera_matrix)

distort_params = [[-4.02049257e-01, 2.06995701e-01, -3.94682806e-04, -1.01677466e-03, -5.42424167e-02]]
distort_params = np.array(distort_params)

# ===================================== REMOVING DISTORTION ============================================ #
file_path = r"C:\Users\CHAND\PycharmProjects\Object_Tracking_along_with_Speed_and_Distance_Estimation\Resources" \
            r"\camera_calibration\PH20230828-104159-000070.JPG"
frameSize = (1240,720)

sample_image = cv2.imread(file_path)
h, w = sample_image.shape[:2]

new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distort_params, (w, h), 1, (w, h))


# # ---------------- Method 1 to undistort the image
#
# dst = cv2.undistort(sample_image, camera_matrix, distort_params, None, new_camera_mtx)
# # cropping the dst image using roi
# x,y,w,h = roi
# cropped_dst = dst[y:y+h, x:x+w]
# # Resizing the undistorted image and Displaying it and its cropped version
# dst = cv2.resize(dst, frameSize)
# cv2.imshow("undistorted image", dst)
# cv2.imshow("cropped image", cropped_dst)
# cv2.waitKey(0)


# ------------------- Method 1 to undistort the image (using Remapping)

map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distort_params, None, new_camera_mtx, (w,h), 5)
dst_2 = cv2.remap(sample_image, map_x, map_y, cv2.INTER_LINEAR)
# cropping the dst image using roi
x,y,w,h = roi
cropped_dst_2 = dst_2[y:y+h, x:x+w]
# Resizing the undistorted image and Displaying it and its cropped version
dst_2 = cv2.resize(dst_2, frameSize)
cv2.imshow("undistorted image", dst_2)
cv2.imshow("cropped image", cropped_dst_2)
cv2.waitKey(0)






