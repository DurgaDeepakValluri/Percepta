import numpy as np
import cv2
from percepta.global_utils.motion_utils import compute_essential_matrix, recover_pose, triangulate_points
from percepta.global_utils.input_utils import load_frames
from percepta.global_utils.log_utils import initialize_logger


class VisualOdometry:
    """
    VisualOdometry: A beginner-friendly monocular visual odometry pipeline.

    Features:
    - Simple one-line interface for running the pipeline.
    - Motion estimation using Essential Matrix and Pose Recovery.
    - 3D point reconstruction using triangulation.
    """

    def __init__(self, camera_type="monocular", camera_matrix=None):
        """
        Initialize the VisualOdometry pipeline.

        Parameters:
        - camera_type (str): Type of camera ("monocular", "stereo", or "rgbd").
        - camera_matrix (np.ndarray): Camera intrinsic matrix (3x3). If None, an identity matrix is used.
        """
        self.logger = initialize_logger(self.__class__.__name__)
        self.logger.info("Initializing VisualOdometry...")

        # Validate camera type
        self.camera_type = self._validate_camera_type(camera_type)

        # Validate or set default camera matrix
        self.K = self._validate_camera_matrix(camera_matrix)

        # Internal state
        self.trajectory = [[0, 0, 0]]  # Initial position
        self.pointcloud = []  # List of 3D points

    def _validate_camera_type(self, camera_type):
        """
        Validate the camera type.

        Parameters:
        - camera_type (str): Type of camera ("monocular", "stereo", or "rgbd").

        Returns:
        - camera_type (str): Validated camera type.
        """
        valid_camera_types = ["monocular", "stereo", "rgbd"]
        if camera_type not in valid_camera_types:
            raise ValueError(f"Invalid camera type '{camera_type}'. Must be one of {valid_camera_types}.")
        self.logger.info(f"Using {camera_type} camera.")
        return camera_type

    def _validate_camera_matrix(self, camera_matrix):
        """
        Validate the camera intrinsic matrix.

        Parameters:
        - camera_matrix (np.ndarray): Camera intrinsic matrix (3x3).

        Returns:
        - camera_matrix (np.ndarray): Validated camera matrix.
        """
        if camera_matrix is None:
            self.logger.warning(
                "No camera intrinsics provided. Using identity matrix. "
                "This disables real-world scale and may cause unstable 3D reconstruction."
            )
            return np.eye(3, dtype=np.float64)
        if not isinstance(camera_matrix, np.ndarray) or camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be a 3x3 numpy array.")
        return camera_matrix.astype(np.float64)

    def run(self, input_source):
        """
        Run the VisualOdometry pipeline on the input source.

        Parameters:
        - input_source: Path to a video file, folder of images, or webcam index.

        Returns:
        - trajectory (list): List of 3D positions (trajectory).
        - pointcloud (list): List of 3D points reconstructed from the scene.
        """
        self.logger.info("Loading frames...")
        frames = load_frames(input_source, logger=self.logger)
        if len(frames) < 2:
            raise ValueError("At least 2 frames are required to run VisualOdometry.")

        self.logger.info(f"Loaded {len(frames)} frames. Starting pipeline...")

        for i in range(len(frames) - 1):
            frame1, frame2 = frames[i], frames[i + 1]
            self._process_frame_pair(frame1, frame2)

        self.logger.info("Pipeline completed.")
        return self.trajectory, self.pointcloud

    def _process_frame_pair(self, frame1, frame2):
        """
        Process a pair of frames to estimate motion and reconstruct 3D points.

        Parameters:
        - frame1 (np.ndarray): First frame.
        - frame2 (np.ndarray): Second frame.
        """
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        # Detect and compute keypoints and descriptors
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.logger.warning("Too few keypoints detected. Skipping frame pair.")
            return

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            self.logger.warning("Not enough matches found. Skipping frame pair.")
            return

        # Extract matched points
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Compute essential matrix
        E = compute_essential_matrix(points1, points2, self.K)
        if E is None:
            self.logger.warning("Failed to compute essential matrix. Skipping frame pair.")
            return

        # Recover pose
        R, t = recover_pose(E, points1, points2, self.K)
        if np.isnan(R).any() or np.linalg.norm(t) > 10:
            self.logger.warning("Unstable pose detected. Skipping frame pair.")
            return

        # Update trajectory
        last_position = np.array(self.trajectory[-1])
        new_position = last_position + t.flatten()
        self.trajectory.append(new_position.tolist())

        # Triangulate points
        proj1 = self.K @ np.eye(3, 4)
        proj2 = self.K @ np.hstack((R, t))
        points_3d = triangulate_points(points1, points2, proj1, proj2)

        # Update point cloud
        self.pointcloud.extend(points_3d.tolist())

    def get_trajectory(self):
        """
        Get the estimated trajectory.

        Returns:
        - trajectory (list): List of 3D positions.
        """
        return self.trajectory

    def get_pointcloud(self):
        """
        Get the reconstructed 3D point cloud.

        Returns:
        - pointcloud (list): List of 3D points.
        """
        return self.pointcloud