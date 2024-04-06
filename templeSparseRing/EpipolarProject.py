import numpy as np
import os
import cv2
import open3d as o3d

# calibrates the camera and returns the undistorted image 
# results are more accurate if several images of the pattern are used 
# if only one image is used n_frame = 1
# src_dir - source directory with the caibration images
# if empty images are grabbed from camera 
def calibrate(n_frames, src_dir='', img_template='image_%d.jpg'):
    print('Calibrating on {} images'.format(n_frames))

    if src_dir:
        print('Capturing images from {}'.format(src_dir))
        # src_fname = os.path.join('./a5-tracking/data/'+seq_name, img_template)
        cam_path = os.path.join(src_dir, img_template)
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
            # cv2.drawChessboardCorners(img, (6,7), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            img_id += 1

        else:
            print('Corner detection failed in image {}'.format(img_id))
                
    if not images:
        raise IOError('No valid images found for calibration')

    # calibrate camera from obj_points and img_points
    _, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_gs.shape[::-1], None, None)
    
    # get optimal params of undistorsed images 
    h, w = img_gs.shape[:2]

    refined_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

    cv2.destroyAllWindows() 

    return refined_K, dist, images

def find_matches(imgL, imgR):
    # Convert images to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    key_pointL, desL = orb.detectAndCompute(grayL, None)
    key_pointR, desR = orb.detectAndCompute(grayR, None)

    matches = bf.match(desL, desR)
    matches = sorted(matches, key=lambda x: x.distance)
    return key_pointL, key_pointR, matches

def estimate_Essential_Matrix(key_pointsL, key_pointsR, matches, K):
    pts1 = np.float32([key_pointsL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([key_pointsR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return pts1, pts2, E, R, t

def triangulate_points(pts1, pts2, R, t, K):
    # Build the projection matrices for the two cameras
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # Convert the projection matrices to the camera coordinate system
    P1 = K @ P1
    P2 = K @ P2

    # Triangulate the 3D points
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
    points_3D = points_3D[:3, :].T
    return points_3D

def create_point_cloud(points_3d):
    """
    Creates a point cloud from 3D points.

    Args:
        your_3d_points (numpy.ndarray): Nx3 array of 3D coordinates.

    Returns:
        open3d.geometry.PointCloud: The created point cloud.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    return point_cloud

def read_images(n_frames, src_dir, img_template):
    if src_dir:
        print('Capturing images from {}'.format(src_dir))
        # src_fname = os.path.join('./a5-tracking/data/'+seq_name, img_template)
        cam_path = os.path.join(src_dir, img_template)
        cap = cv2.VideoCapture(cam_path)
    else:
        print('Capturing images from camera')
        cap = cv2.VideoCapture(0)

    images = []

    img_id = 0
    while img_id < n_frames:

        ret, img = cap.read()
        if ret:     
            # img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow(f"img {img_id}", img)
            # cv2.waitKey(500)
            images.append(img)
        else:
            print('Capture of Image {} was unsuccessful'.format(img_id + 1))
            break
    return images

def visualize_point_cloud(points_array):
    """
    Visualizes a point cloud using Open3D.

    Args:
        point_cloud (open3d.geometry.PointCloud): The point cloud to visualize.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)
    o3d.visualization.draw_geometries([point_cloud])

def create_point_cloud(your_3d_points):
    """
    Creates a point cloud from 3D points.

    Args:
        your_3d_points (numpy.ndarray): Nx3 array of 3D coordinates.

    Returns:
        open3d.geometry.PointCloud: The created point cloud.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(your_3d_points)
    return point_cloud

if __name__ == "__main__":
    src_dir = 'cv_chessboard'
    n_frames = 13
    img_template = 'image_%d.jpg' 
 
    # calibrate camera from n_frames 
    K, dist, images = calibrate(n_frames, src_dir, img_template)

    images = read_images(15, 'templeSparseRing', 'templeSR%04d.png')
    obj_3d = np.array([])
    for i in range(len(images)-1):
        i = 0
        key_pointL, key_pointR, matches = find_matches(images[i], images[i+1])
        pts1, pts2, E, R, t = estimate_Essential_Matrix(key_pointL, key_pointR, matches, K)
        points_3D = triangulate_points(pts1, pts2, R, t, K)
        print(obj_3d.size)

        if obj_3d.size == 0:
            obj_3d = points_3D
        else:
            obj_3d = np.concatenate((obj_3d, points_3D), axis=0)
    # # print(obj_3d)
    # # point_cloud = create_point_cloud(obj_3d)
    visualize_point_cloud(obj_3d)