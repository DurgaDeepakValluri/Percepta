import numpy as np
import open3d as o3d
import cv2
from percepta.vo._base_vo import BaseVO
from percepta.global_utils.motion_utils import compute_essential_matrix, recover_pose, triangulate_points


class StereoVO(BaseVO):
    """
    StereoVO: A modular stereo visual odometry pipeline.

    Supports:
    - Stereo feature-based motion estimation
    - 3D point triangulation using stereo geometry
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
        baseline=0.1,  # Baseline distance between stereo cameras
        frame_stride=1,
        parallel=False,
        input_size=None,
        config=None
    ):
        """
        Initialize StereoVO pipeline.

        Parameters:
        - detector (str): Feature detector to use.
        - matcher (str): Feature matcher to use.
        - viz (dict): Toggle visuals individually.
        - viz_only (list): Visuals to enable exclusively.
        - output_types (list): What outputs to generate.
        - camera_matrix (ndarray): Intrinsic matrix (3x3).
        - baseline (float): Baseline distance between stereo cameras.
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
        self.baseline = baseline
        self.logger.info(f"StereoVO initialized with baseline: {self.baseline} meters.")

    def _process_frame_pair(self, pair):
        """
        Process a pair of stereo frames from the input list.

        Parameters:
        - pair: Tuple containing (index, left_frame, right_frame).
        """
        i, left_frame, right_frame = pair
        points_3d = self.process_stereo_frame(left_frame, right_frame)
        if points_3d is not None:
            self._update_pointcloud(points_3d)

    def process_stereo_frame(self, left_frame, right_frame):
        """
        Process a stereo frame pair:
        - Detect and match keypoints
        - Triangulate 3D points using stereo geometry

        Parameters:
        - left_frame (np.ndarray): Left stereo frame.
        - right_frame (np.ndarray): Right stereo frame.

        Returns:
        - points_3d (np.ndarray): Nx3 array of 3D points.
        """
        left_frame = self._preprocess_frame(left_frame)
        right_frame = self._preprocess_frame(right_frame)

        kp1, des1 = self.detector.detectAndCompute(left_frame, None)
        kp2, des2 = self.detector.detectAndCompute(right_frame, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.logger.warning("Too few keypoints in stereo frames.")
            return None

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            self.logger.warning("Not enough matches in stereo frames.")
            return None

        if self.viz.get('matches', False):
            self.visualize_matches(left_frame, right_frame, kp1, kp2, matches)

        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        proj_matrix1 = self.K @ np.eye(3, 4)
        proj_matrix2 = self.K @ np.array([[1, 0, 0, -self.baseline]]).T

        points_3d = triangulate_points(points1, points2, proj_matrix1, proj_matrix2)
        return points_3d

    def visualize_matches(self, left_frame, right_frame, kp1, kp2, matches, limit=50):
        """
        Visualize feature matches between stereo frames using OpenCV.
        """
        f1 = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR) if len(left_frame.shape) == 2 else left_frame
        f2 = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR) if len(right_frame.shape) == 2 else right_frame
        match_img = cv2.drawMatches(f1, kp1, f2, kp2, matches[:limit], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Stereo Feature Matches", match_img)
        cv2.waitKey(1)