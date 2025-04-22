import logging
import cv2
import numpy as np
import open3d as o3d
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

class MonoVO:
 
    """
    MonoVO: A modular monocular visual odometry pipeline.
    
    Supports:
    - Classic feature-based motion estimation
    - Multithreading for speed
    - Camera intrinsic matrix input
    - Output saving to disk
    - Real-time and post-run visualization
    """    
 
    VALID_DETECTORS = ['ORB', 'SIFT', 'SURF']
    VALID_MATCHERS = ['BF', 'FLANN']
    VALID_VIZ_KEYS = ['matches', 'keypoints', 'trajectory', 'pointcloud']
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
        input_size=None
    ):
        
        """
        Initialize MonoVO pipeline.

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
        """

        self._initialize_logging()
        self._validate_inputs(detector, matcher, viz, viz_only, output_types)

        self.detector_name = detector.upper()
        self.matcher_name = matcher.upper()
        self.output_types = output_types or ['trajectory', 'pointcloud', 'matches']
        self._configure_visualizations(viz, viz_only)

        self.detector = self._initialize_detector(self.detector_name)
        self.matcher = self._initialize_matcher(self.detector_name, self.matcher_name)

        self.K = camera_matrix if camera_matrix is not None else np.eye(3)
        self.frame_stride = frame_stride
        self.parallel = parallel
        self.input_size = input_size

        self.trajectory = [[0, 0, 0]]
        self.pointcloud = o3d.geometry.PointCloud()
        self.matches = []

        self.logger.info("MonoVO pipeline initialized successfully.")

    def _initialize_logging(self):
        """Configure the logger for MonoVO."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MonoVO pipeline...")

    def _validate_inputs(self, detector, matcher, viz, viz_only, output_types):
        """Check for invalid parameters and raise errors."""
        if detector.upper() not in self.VALID_DETECTORS:
            raise ValueError(f"Invalid detector: {detector}. Choose from {self.VALID_DETECTORS}.")
        if matcher.upper() not in self.VALID_MATCHERS:
            raise ValueError(f"Invalid matcher: {matcher}. Choose from {self.VALID_MATCHERS}.")
        if viz:
            for key in viz:
                if key not in self.VALID_VIZ_KEYS:
                    raise ValueError(f"Invalid viz key: {key}. Choose from {self.VALID_VIZ_KEYS}.")
        if viz_only:
            for key in viz_only:
                if key not in self.VALID_VIZ_KEYS:
                    raise ValueError(f"Invalid viz_only key: {key}. Choose from {self.VALID_VIZ_KEYS}.")
        if output_types:
            for o in output_types:
                if o not in self.VALID_OUTPUTS:
                    raise ValueError(f"Invalid output: {o}. Choose from {self.VALID_OUTPUTS}.")

    def _configure_visualizations(self, viz, viz_only):
        """Configure visualizations from viz and viz_only settings."""
        default_viz = {k: True for k in self.VALID_VIZ_KEYS}
        if viz_only:
            self.viz = {k: k in viz_only for k in default_viz}
        else:
            self.viz = default_viz.copy()
            if viz:
                self.viz.update({k: bool(v) for k, v in viz.items() if k in self.viz})
        self.logger.info(f"Visualization settings: {self.viz}")

    def _initialize_detector(self, name):
         """Return initialized OpenCV detector based on name."""
        detectors = {
            'ORB': cv2.ORB_create,
            'SIFT': cv2.SIFT_create,
            'SURF': cv2.xfeatures2d.SURF_create,
        }
        return detectors[name]()

    def _initialize_matcher(self, detector, matcher):
         """Return initialized OpenCV matcher based on detector and matcher."""
        if matcher == 'BF':
            norm = cv2.NORM_HAMMING if detector == 'ORB' else cv2.NORM_L2
            return cv2.BFMatcher(norm, crossCheck=True)
        elif matcher == 'FLANN':
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)

    def _preprocess_frame(self, frame):
        
         """
        Standardize the input frame:
        - Convert to grayscale if needed
        - Resize if input_size is set
        - Convert to uint8 if necessary
        """

        if frame is None:
            raise ValueError("Received a None frame.")
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.input_size is not None:
            frame = cv2.resize(frame, self.input_size)
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8) if frame.dtype in [np.float32, np.float64] else frame.astype(np.uint8)
        return frame

    def visualize_matches(self, frame1, frame2, kp1, kp2, matches, limit=50):
        """Visualize feature matches between two frames using OpenCV.
        Shown live in an OpenCV window"""
        f1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR) if len(frame1.shape) == 2 else frame1
        f2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR) if len(frame2.shape) == 2 else frame2
        match_img = cv2.drawMatches(f1, kp1, f2, kp2, matches[:limit], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Feature Matches", match_img)
        cv2.waitKey(1)

    def visualize_trajectory(self):
        """
        Plot X-Z camera trajectory using matplotlib.
        Only works if trajectory has >1 point.
        """
        if 'trajectory' not in self.output_types or len(self.trajectory) < 2:
            self.logger.warning("No trajectory to visualize.")
            return
        traj = np.array(self.trajectory)
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 2], marker='o')
        plt.title("Camera Trajectory (X-Z plane)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.grid()
        plt.axis('equal')
        plt.show()

    def visualize_pointcloud(self):
        """
        Visualize current 3D point cloud using Open3D.
        Skips if no points are available.
        """
        if self.pointcloud.is_empty():
            self.logger.warning("Point cloud is empty. Nothing to visualize.")
            return
        self.logger.info(f"Showing point cloud with {len(self.pointcloud.points)} points.")
        o3d.visualization.draw_geometries([self.pointcloud])

    def _process_frame_pair(self, pair):
        """
        Process a pair of frames from the input list.
        Used in both serial and parallel modes.
        """
        i, frame1, frame2 = pair
        R, t, points_3d = self.process_frame(frame1, frame2)
        if R is not None and t is not None:
            self._update_trajectory(t)
        if points_3d is not None:
            self._update_pointcloud(points_3d)

    def process_frame(self, frame1, frame2):
          """
        Main pipeline step:
        - Detect and match keypoints
        - Estimate essential matrix
        - Recover pose
        - Triangulate points
        - Optionally visualize results
        Returns:
        - R: Rotation matrix
        - t: Translation vector
        - points_3d: Nx3 array of 3D points
        """
        frame1 = self._preprocess_frame(frame1)
        frame2 = self._preprocess_frame(frame2)

        kp1, des1 = self.detector.detectAndCompute(frame1, None)
        kp2, des2 = self.detector.detectAndCompute(frame2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.logger.warning("Too few keypoints.")
            return None, None, None

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            self.logger.warning("Not enough matches.")
            return None, None, None

        if self.viz.get('matches', False):
            self.visualize_matches(frame1, frame2, kp1, kp2, matches)

        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None

        _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=self.K)
        if np.isnan(R).any() or np.linalg.norm(t) > 10:
            self.logger.warning("Unstable pose.")
            return None, None, None

        proj1 = self.K @ np.eye(3, 4)
        proj2 = self.K @ np.hstack((R, t))
        points_3d = cv2.triangulatePoints(proj1, proj2, points1.T, points2.T)
        points_3d /= points_3d[3]

        if 'matches' in self.output_types:
            self.matches.append((kp1, kp2, matches))

        return R, t, points_3d[:3].T

    def run_pipeline(self, frames):
        """
        Run the VO pipeline on a list of image frames.
        Skips frames if frame_stride > 1.
        Can run in parallel if enabled.
        """
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("Input must be a list of at least 2 frames")

        frame_pairs = [
            (i, frames[i], frames[i + 1])
            for i in range(0, len(frames) - 1, self.frame_stride)
        ]

        if self.parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(self._process_frame_pair, frame_pairs)
        else:
            for pair in frame_pairs:
                self._process_frame_pair(pair)

        self.logger.info("Pipeline finished.")

    def _update_trajectory(self, t):
         """Add new position to the trajectory list."""
        last = np.array(self.trajectory[-1])
        new_pos = last + t.flatten()
        self.trajectory.append(new_pos.tolist())

    def _update_pointcloud(self, points_3d):
        """Append new 3D points to the point cloud."""
        try:
            new_pts = o3d.utility.Vector3dVector(points_3d)
            self.pointcloud += o3d.geometry.PointCloud(points=new_pts)
        except Exception as e:
            self.logger.warning(f"Pointcloud update failed: {e}")

    def save_outputs(self, output_dir="outputs"):
        """
        Save results to disk:
        - trajectory.csv
        - pointcloud.ply
        """
        os.makedirs(output_dir, exist_ok=True)
        if 'trajectory' in self.output_types:
            np.savetxt(Path(output_dir) / "trajectory.csv", np.array(self.trajectory), delimiter=",")
        if 'pointcloud' in self.output_types and len(self.pointcloud.points) > 0:
            o3d.io.write_point_cloud(str(Path(output_dir) / "pointcloud.ply"), self.pointcloud)
        self.logger.info(f"Outputs saved to {output_dir}/")

    def get_output(self):
        """
        Return a dictionary of outputs:
        - 'trajectory': List of poses
        - 'pointcloud': Open3D pointcloud
        - 'matches': Keypoint matches
        """
        return {
            
            'trajectory': self.trajectory.copy() if 'trajectory' in self.output_types else None,
            'pointcloud': self.pointcloud if 'pointcloud' in self.output_types else None,
            'matches': self.matches.copy() if 'matches' in self.output_types else None,
        }

    def reset(self):
        """
        Reset internal state.
        Clears trajectory, matches, and pointcloud.
        """
        self.trajectory = [[0, 0, 0]]
        self.pointcloud.clear()
        self.matches.clear()
        self.logger.info("Pipeline reset.")
