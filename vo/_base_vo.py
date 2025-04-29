import os
import time
import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from percepta.global_utils.input_utils import load_frames
from percepta.global_utils.feature_utils import initialize_detector, initialize_matcher
from percepta.global_utils.image_standardizer import preprocess_frame
from percepta.global_utils.trajectory_utils import update_trajectory, save_trajectory
from percepta.global_utils.pointcloud_utils import update_pointcloud, save_pointcloud
from percepta.global_utils.timing_utils import start_timer, end_timer, compute_fps
from percepta.global_utils.log_utils import initialize_logger
from percepta.vo.vo_utils.imu_utils import fuse_imu_with_vo
from percepta.config import Config
from percepta.vo.metrics.vo_metrics import (  # Importing metrics
    compute_absolute_trajectory_error,
    compute_relative_pose_error,
    compute_scale_drift,
    compute_reprojection_error,
)





class BaseVO:
    """
    BaseVO: Foundation class for all Visual Odometry types (Mono, Stereo, RGBD).

    Handles:
    - Detector and matcher initialization
    - Input validation and preprocessing
    - Frame standardization
    - Common pipeline infrastructure (frame skipping, parallelism, logging)
    """

    VALID_OUTPUTS = ['trajectory', 'pointcloud', 'matches']

    def __init__(
        self,
        detector='ORB',
        matcher='BF',
        viz=None,
        viz_only=None,
        output_types=None,
        camera_matrix=None,
        frame_stride=1,
        parallel=False,
        input_size=None,
        use_imu=False,
        config=None
    ):
        """
        Initialize BaseVO pipeline settings.

        Parameters:
        - detector (str): Feature detector to use.
        - matcher (str): Feature matcher to use.
        - viz (dict): Toggle visuals individually.
        - viz_only (list): Visuals to enable exclusively.
        - output_types (list): What outputs to generate.
        - camera_matrix (ndarray): Intrinsic matrix (3x3).
        - frame_stride (int): Frame skipping factor.
        - parallel (bool): Enable multi-threaded processing.
        - input_size (tuple): Resize input images to (w, h).
        - use_imu (bool): Whether to enable IMU integration for VIO.
        - config (Config): Optional configuration object for advanced settings.
        """
        self.logger = initialize_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        self.config = config or Config()
        self.detector = initialize_detector(detector)
        self.matcher = initialize_matcher(detector, matcher)

        self.K = self._validate_camera_matrix(camera_matrix)
        self.frame_stride = frame_stride
        self.parallel = parallel
        self.input_size = input_size
        self.use_imu = use_imu

        self.trajectory = [[0, 0, 0]]
        self.pointcloud = o3d.geometry.PointCloud()
        self.matches = []

        self.lock = Lock()
        self.logger.info(f"{self.__class__.__name__} initialized successfully.")

    def _validate_camera_matrix(self, camera_matrix):
        """
        Validate the camera intrinsic matrix.
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

    def run_pipeline(self, input_source):
        """
        Run the visual odometry pipeline on the input source.
        """
        start_time = start_timer()
        frames = load_frames(input_source, preprocess_func=lambda f: preprocess_frame(f, self.input_size), logger=self.logger)
        if len(frames) < 2:
            raise ValueError("At least 2 frames are required to run VO.")

        frame_pairs = [(i, frames[i], frames[i + 1]) for i in range(0, len(frames) - 1, self.frame_stride)]

        if self.parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(self._process_frame_pair, frame_pairs)
        else:
            for pair in frame_pairs:
                self._process_frame_pair(pair)

        elapsed_time = end_timer(start_time)
        fps = compute_fps(start_time, len(frame_pairs))
        self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds at {fps:.2f} FPS.")

    def _process_frame_pair(self, frame_pair):
        """
        Abstract method for processing a pair of frames.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")

    def _update_trajectory(self, t):
        """
        Update the trajectory with a new translation vector.
        """
        with self.lock:
            self.trajectory = update_trajectory(self.trajectory, t.flatten())

    def _update_pointcloud(self, points_3d):
        """
        Update the point cloud with new 3D points.
        """
        with self.lock:
            self.pointcloud = update_pointcloud(self.pointcloud, points_3d)

    def save_outputs(self, output_dir="outputs"):
        """
        Save trajectory and point cloud to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        if 'trajectory' in self.output_types:
            save_trajectory(self.trajectory, os.path.join(output_dir, "trajectory.csv"))
        if 'pointcloud' in self.output_types:
            save_pointcloud(self.pointcloud, os.path.join(output_dir, "pointcloud.ply"))
        self.logger.info(f"Outputs saved to {output_dir}/")

    def reset(self):
        """
        Reset internal state.
        """
        self.trajectory = [[0, 0, 0]]
        self.pointcloud.clear()
        self.matches.clear()
        self.logger.info("Pipeline reset.")

    def evaluate(self, ground_truth_trajectory=None, ground_truth_points=None):
        """
        Evaluate the VO pipeline using ground truth data.

        Parameters:
        - ground_truth_trajectory (list): List of 3D positions (Nx3) from the ground truth trajectory.
        - ground_truth_points (list): List of 3D points (Nx3) from the ground truth point cloud.

        Returns:
        - metrics (dict): Dictionary of evaluation metrics.
        """
        
        metrics = {}

        # Evaluate trajectory metrics if ground truth is provided
        if ground_truth_trajectory is not None:
            metrics['ATE'] = compute_absolute_trajectory_error(self.trajectory, ground_truth_trajectory)
            metrics['RPE'] = compute_relative_pose_error(self.trajectory, ground_truth_trajectory)
            metrics['Scale Drift'] = compute_scale_drift(self.trajectory, ground_truth_trajectory)

        # Log the metrics
        self.logger.info(f"Evaluation Metrics: {metrics}")
        return metrics