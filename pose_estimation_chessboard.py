import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'data\chessboard.mp4'
K = np.array([[797.02018203, 0, 643.78922132],
              [0, 799.67012019, 313.95795079],
              [0, 0, 1]])
dist_coeff = np.array([0.02301998,  0.16248255, -0.02179514,  0.00066636, -0.37456221])
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare 3D points for a square pyramid
pyramid_base = board_cellsize * np.array([[3, 2,  0], [5, 2,  0], [5, 4,  0], [3, 4,  0]])
pyramid_top = board_cellsize * np.array([[4, 3, -2]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the pyramid on the image
        pyramid_base_proj, _ = cv.projectPoints(pyramid_base, rvec, tvec, K, dist_coeff)
        pyramid_top_proj, _ = cv.projectPoints(pyramid_top, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(pyramid_base_proj)], True, (255, 0, 0), 2)
        for i in range(4):
            cv.line(img, tuple(np.int32(pyramid_base_proj[i].ravel())), tuple(np.int32(pyramid_top_proj[0].ravel())), (0, 0, 255), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
