import numpy as np
import os
import cv2

# calibrates the camera and returns the undistorted image 
# results are more accurate if several images of the pattern are used 
# if only one image is used n_frame = 1
# src_dir - source directory with the caibration images
# if empty images are grabbed from camera 
def calibrate(n_frames, src_dir='', img_template='image_%d.jpg', vis_delay=500):
    print('Calibrating on {} images'.format(n_frames))

    if src_dir:
        print('Capturing images from {}'.format(src_dir))
        # src_fname = os.path.join('./a5-tracking/data/'+seq_name, img_template)
        cam_path = os.path.join('a6-ar/cv_chessboard', img_template)
        cap = cv2.VideoCapture(cam_path)
    else:
        print('Capturing images from camera')
        cap = cv2.VideoCapture(0)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare 3D object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # depends on how many squares your pattern has 
    # I included the opencv calibration sequence but you should take your own also
    # in the opencv sequence pattern is 6x7
    
    corners_3d = np.zeros((6*7,3), np.float32)
    corners_3d[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2) # generates each point on the 2d plane

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    images = []

    img_id = 0
    while img_id < n_frames:

        ret, img = cap.read()
        if not ret:
            print('Capture of Image {} was unsuccessful'.format(img_id + 1))
            break


        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # use function ret, corners = cv2.findChessboardCorners
        ret, corners = cv2.findChessboardCorners(img_gs, (6,7), None)
        
        # If found, add object points, image points (after refining them)
        if ret:     
            # refine to subpixel with cv2.cornerSubPix
            corners2 = cv2.cornerSubPix(img_gs,corners, (11,11), (-1,-1), criteria)

            obj_points.append(corners_3d)
            img_points.append(corners2)
            images.append(img_gs)

            # Draw and display the corners
            # use drawChessboardCorners
            cv2.drawChessboardCorners(img, (6,7), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            img_id += 1

        else:
            print('Corner detection failed in image {}'.format(img_id))
                
    if not images:
        raise IOError('No valid images found for calibration')

    # calibrate camera from obj_points and img_points
    ret2, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_gs.shape[::-1], None, None)
    
    # get optimal params of undistorsed images 
    h, w = img_gs.shape[:2]

    refined_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

    # display undistorsed images
    for img_id, gray_img in enumerate(images):
        # undistort
        dst = cv2.undistort(gray_img, K, dist, None, refined_K)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        cv2.imshow('undistorted img', dst)
        cv2.waitKey(vis_delay)

    cv2.destroyAllWindows() 

    return refined_K, dist, images
