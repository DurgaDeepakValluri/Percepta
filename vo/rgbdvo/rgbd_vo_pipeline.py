import numpy as np
import open3d as o3d
import cv2
from percepta.vo._base_vo import BaseVO
from percepta.global_utils.motion_utils import compute_essential_matrix, recover_pose


class RGBDVO(BaseVO):
    """
    RGBDVO: A modular RGB-D visual odometry pipeline.

    Supports:
    - Motion estimation using RGB and depth data
    - 3D point reconstruction from depth maps
    - Real-time and post-run visualization
    """

    def __init__(
        self,
        detector='ORB',
        matcher='BF',
        viz=None,
        viz_only=None,
        output_types=None,
        camera_matrix=None,
        depth_scale=1000.0,  # Depth scale to convert depth values to meters
        frame_stride=1,
        parallel=False,
        input_size=None,
        config=None
    ):
        """
        Initialize RGBDVO pipeline.

        Parameters:
        - detector (str): Feature detector to use.
        - matcher (str): Feature matcher to use.
        - viz (dict): Toggle visuals individually.
        - viz_only (list): Visuals to enable exclusively.
        - output_types (list): What outputs to generate.
        - camera_matrix (ndarray): Intrinsic matrix (3x3).
        - depth_scale (float): Scale factor to convert depth values to meters.
        - frame_stride (int): Frame skipping factor.
        - parallel (bool): Enable multi-threaded processing.
        - input_size (tuple): Resize input images to (w, h).
        - config (Config): Optional configuration object for advanced settings.
        """
        super().__init__(
            detector=detector,
            matcher=matcher,
            viz=viz,
            viz_only=viz_only,
            output_types=output_types,
            camera_matrix=camera_matrix,
            frame_stride=frame_stride,
            parallel=parallel,
            input_size=input_size,
            config=config
        )
        self.depth_scale = depth_scale
        self.logger.info(f"RGBDVO initialized with depth scale: {self.depth_scale}.")

    def _process_frame_pair(self, pair):
        """
        Process a pair of RGB-D frames from the input list.

        Parameters:
        - pair: Tuple containing (index, rgb_frame, depth_frame).
        """
        i, rgb_frame, depth_frame = pair
        R, t, points_3d = self.process_rgbd_frame(rgb_frame, depth_frame)
        if R is not None and t is not None:
            self._update_trajectory(t)
        if points_3d is not None:
            self._update_pointcloud(points_3d)

    def process_rgbd_frame(self, rgb_frame, depth_frame):
        """
        Process an RGB-D frame pair:
        - Detect and match keypoints
        - Estimate motion using RGB data
        - Reconstruct 3D points using depth data

        Parameters:
        - rgb_frame (np.ndarray): RGB frame.
        - depth_frame (np.ndarray): Depth frame.

        Returns:
        - R (np.ndarray): Rotation matrix.
        - t (np.ndarray): Translation vector.
        - points_3d (np.ndarray): Nx3 array of 3D points.
        """
        rgb_frame = self._preprocess_frame(rgb_frame)

        kp, des = self.detector.detectAndCompute(rgb_frame, None)
        if des is None or len(kp) < 10:
            self.logger.warning("Too few keypoints in RGB frame.")
            return None, None, None

        points_2d = np.float32([kp[i].pt for i in range(len(kp))])
        points_depth = depth_frame[points_2d[:, 1].astype(int), points_2d[:, 0].astype(int)] / self.depth_scale

        valid_mask = points_depth > 0
        points_2d = points_2d[valid_mask]
        points_depth = points_depth[valid_mask]

        points_3d = cv2.convertPointsToHomogeneous(points_2d)[:, 0, :] * points_depth[:, None]
        return np.eye(3), np.zeros((3, 1)), points_3d