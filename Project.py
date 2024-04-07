import numpy as np
import os
import cv2
import open3d as o3d

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

    # # display undistorsed images
    # for img_id, gray_img in enumerate(images):
    #     # undistort
    #     dst = cv2.undistort(gray_img, K, dist, None, refined_K)

    #     # crop the image
    #     x, y, w, h = roi
    #     dst = dst[y:y + h, x:x + w]

    #     cv2.imshow('undistorted img', dst)
    #     cv2.waitKey(vis_delay)

    cv2.destroyAllWindows() 

    return refined_K, dist, images

def detectAndDescribe(image):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # get kps and features
    kps, features = sift.detectAndCompute(image, None)
    return kps, features


def ratio_distance(des1,des2, ratio):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test and save good matches in a list of DMatch elements
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    # print(good)
    # good_match = sorted(good, key = lambda x:x[0].distance)
    return good

# matches, H  = matchKeypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.75)
# input : two sets of keypoints and features as detected by detectAndDescribe
#          match_type = 0 brute force matching
#          match_type = 1 ration distance using param ration
# output : matches - a list of matched of indeces of matched features 
#          (like the one returned by cv2.DescriptorMatcher class)
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75):
    # calculate best matches
    good_matches = ratio_distance(featuresA, featuresB, ratio)  # ratio distance

    # compute homography matrix
    src_pts = np.float32([ kpsA[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpsB[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    return good_matches, H

def triangulate_points(cam0, cam1, pointsL, pointsR):
    # Assuming you have cam0 and cam1 matrices, and pointsL, pointsR
    pnts3D_homogeneous = cv2.triangulatePoints(cam0, cam1, pointsL, pointsR)

    # Convert to 3D points (remove the homogeneous coordinate)
    pnts3D = pnts3D_homogeneous[:3, :] / pnts3D_homogeneous[3, :]

    return pnts3D

def match_keypoints(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matching points
    matching_points1 = [keypoints1[match.queryIdx].pt for match in matches]
    matching_points2 = [keypoints2[match.trainIdx].pt for match in matches]

    return matching_points1, matching_points2

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

def create_mesh(point_cloud, depth):
    """
    Creates a mesh from a point cloud using Poisson surface reconstruction.

    Args:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        depth (int): Depth of the octree used for reconstruction (higher values yield more detailed meshes).

    Returns:
        open3d.geometry.TriangleMesh: The reconstructed mesh.
        densities (numpy.ndarray): Density values associated with each point in the point cloud.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    return mesh, densities

def visualize_point_cloud(point_cloud):
    """
    Visualizes a point cloud using Open3D.

    Args:
        point_cloud (open3d.geometry.PointCloud): The point cloud to visualize.
    """
    o3d.visualization.draw_geometries([point_cloud])

def idk_probably_main(cal_img, obj_img):
    src_dir = 'cv_chessboard'
    n_frames = 13
    img_template = 'image_%d.jpg' 
 
    # calibrate camera from n_frames 
    K, dist, images = calibrate(n_frames, src_dir, img_template)
    # k, d, img = calibrate(10, "", "", 10)
    for i in range(len(obj_img)):
        key_points1, features1 = detectAndDescribe(obj_img[i])
        key_points2, features2 = detectAndDescribe(obj_img[i+1])
        gmatches, H = matchKeypoints(key_points1, key_points2, features1, features2)
        matches1, matches2 = match_keypoints(obj_img[i], obj_img[i+1])
        if len(matches1) == len(matches2):
            points3d = np.array([])
            for i in range(len(matches1)):
                pt3d = triangulate_points(k,k,matches1[i],matches2[i])
                points3d = np.append(points3d, pt3d)
            point_cloud =  create_point_cloud(points3d)
            visualize_point_cloud(point_cloud)



