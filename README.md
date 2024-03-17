# Cmpt470_3DModelProject
Attempting to reconstruct 3d models from 2d images using point matching and triangulation

https://noobtomaster.com/opencv-using-python/performing-3d-reconstruction-from-multiple-images/

# The Process of 3D Reconstruction
The process of 3D reconstruction involves capturing multiple images of an object or scene from different viewpoints and then using these images to infer the 3D structure. This can be achieved by following these general steps:

##1. Image Acquisition
Using a camera or multiple cameras, capture a series of images from different positions around the object or scene. It is important to have a sufficient overlap between these images to ensure accurate reconstruction.

##2. Camera Calibration
Before starting the reconstruction process, it is necessary to calibrate the cameras used for image acquisition. Camera calibration determines the intrinsic and extrinsic parameters of the camera, such as focal length, distortion, and camera pose. OpenCV provides functions to perform camera calibration, improving the accuracy of the reconstruction process.

##3. Feature Extraction and Matching
In this step, features (such as corners, edges, or keypoints) are extracted from the acquired images. These features provide distinctive information about the object or scene and act as reference points for matching corresponding features in different images. OpenCV's feature extraction and matching algorithms simplify this process and help in identifying corresponding points accurately.

##4. Pose Estimation
Using the matched feature points, the relative camera poses for each image pair can be estimated. This estimation determines the transformation between the camera coordinate systems, allowing the creation of a global coordinate system. OpenCV provides methods for pose estimation, including algorithms like RANSAC for robust estimation.

##5. Triangulation
Triangulation is the process of reconstructing 3D points by intersecting corresponding rays from multiple camera viewpoints. OpenCV's triangulation methods use the estimated camera poses and matched feature points to generate a dense 3D point cloud representing the object or scene.

##6. Point Cloud Refinement
The initial point cloud obtained from triangulation often contains noisy or inaccurate points. To improve the quality of the reconstruction, point cloud refinement techniques are employed. These techniques remove outliers, smooth the surface, and reduce noise, resulting in a more accurate 3D representation.

##7. Surface Reconstruction
To obtain a complete 3D model, a surface must be reconstructed from the point cloud. Different algorithms can be employed, such as the Poisson surface reconstruction or Marching Cubes algorithm, to convert the point cloud into a continuous 3D surface mesh.

##8. Texture Mapping and Visualization
Once the 3D surface mesh is obtained, texture mapping techniques can be used to project the acquired images onto the mesh, giving it a realistic appearance. OpenCV enables the process of texture mapping and provides tools for visualization and further analysis of the reconstructed 3D model.
