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
    
    corners_3d = np.zeros((8*6,3), np.float32)
    corners_3d[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) # generates each point on the 2d plane

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
        ret, corners = cv2.findChessboardCorners(img_gs, (8,6), None)
        
        # If found, add object points, image points (after refining them)
        if ret:     
            # refine to subpixel with cv2.cornerSubPix
            corners2 = cv2.cornerSubPix(img_gs,corners, (11,11), (-1,-1), criteria)

            obj_points.append(corners_3d)
            img_points.append(corners2)
            images.append(img_gs)

            # Draw and display the corners
            # use drawChessboardCorners
            # cv2.drawChessboardCorners(img, (9,7), corners2, ret)
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
    
    # Project 3D points onto the images to get pixel coordinates
    img_points1, _ = cv2.projectPoints(points_3D, np.eye(3), np.zeros((3, 1)), K, distCoeffs=None)
    img_points1 = img_points1.squeeze()

    # print("Triangulated 3D points:", points_3D)
    # print("Image points:", img_points1)

    return points_3D, img_points1
    # return points_3D

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
# Define custom key callbacks for navigation

def visualize_point_cloud(point_cloud):
    """
    Visualizes a point cloud using Open3D.

    Args:
        point_cloud (open3d.geometry.PointCloud): Point cloud to visualize.
    """
    # Create visualizer object
    vis = o3d.visualization.Visualizer()

    # Add point cloud to the visualizer
    vis.create_window()
    vis.add_geometry(point_cloud)

    # Set rendering options
    # render_options = vis.get_render_option()
    # render_options.point_size = 5  # Adjust point size as needed

    # Run visualization
    vis.run()

    # Close the visualizer
    vis.destroy_window()

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

def create_point_cloud_with_colors(points_3d, images, img_points, colors=None):
    """
    Creates a point cloud from 3D points with colors attributed from images.

    Args:
        points_3d (numpy.ndarray): Nx3 array of 3D coordinates.
        images (list): List of images.
        img_points (list): List of 2D pixel coordinates of the 3D points in the images.
        colors (list): List of colors corresponding to the 3D points.

    Returns:
        open3d.geometry.PointCloud: The created point cloud with colors.
    """
    num_images = len(images)
    num_points = points_3d.shape[0]

    if colors is None:
        colors = []
    
    for i in range(num_images -1):
        image = images[i]
        img_point = img_points[i]

        # Sample color from the image at the pixel coordinates
        max_x, max_y = image.shape[1] - 1, image.shape[0] - 1
        img_colors = [image[int(pt[1]), int(pt[0])] if 0 <= pt[0] < max_x and 0 <= pt[1] < max_y else image[max_y-1, max_x-1] for pt in img_point]

        colors.extend(img_colors)

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

if __name__ == "__main__":
    src_dir = 'checkerboard'
    n_frames = 15
    img_template = 'image_%d.jpg' 
 
    # calibrate camera from n_frames 
    K, dist, images = calibrate(n_frames, src_dir, img_template)

    # print(K)
    
    # images = read_images(15, 'templeSparseRing', 'templeSR%04d.png')
    images = read_images(24, 'chest', 'image_%d.jpg')
    # images = read_images(22, 'rock', 'image_%d.jpg')
    obj_3d = np.array([])
    img_points_list = []
    # matches_list = []
    for i in range(len(images) - 1):
        # i = 0
        key_pointL, key_pointR, matches = find_matches(images[i], images[i+1])
        # key_points_list.append(key_pointL)
        # matches_list.append(matches)
        pts1, pts2, E, R, t = estimate_Essential_Matrix(key_pointL, key_pointR, matches, K)
        # points_3D = triangulate_points(pts1, pts2, R, t, K)
        points_3D, img_points = triangulate_points(pts1, pts2, R, t, K)
        img_points_list.append(img_points)
    
        # print(obj_3d.size)

        if obj_3d.size == 0:
            obj_3d = points_3D
        else:
            obj_3d = np.concatenate((obj_3d, points_3D), axis=0)
    # print("Number of images processed:", len(images))
    # print("Number of key points lists:", len(key_points_list))
    # print("Number of matches lists:", len(matches_list))
    # point_cloud_with_colors = create_point_cloud_with_colors(obj_3d, images, key_points_list, matches_list)
    # visualize_point_cloud(point_cloud_with_colors)
    # # print(obj_3d)
    point_cloud = create_point_cloud(obj_3d)
    # point_cloud_with_colors = create_point_cloud_with_colors(obj_3d, images, img_points_list)
    # visualize_point_cloud(point_cloud_with_colors)
    visualize_point_cloud(point_cloud)