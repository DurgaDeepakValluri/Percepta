import numpy as np


def compute_absolute_trajectory_error(estimated_trajectory, ground_truth_trajectory):
    """
    Compute the Absolute Trajectory Error (ATE) between the estimated trajectory and the ground truth.

    Parameters:
    - estimated_trajectory (list): List of 3D positions (Nx3) from the VO pipeline.
    - ground_truth_trajectory (list): List of 3D positions (Nx3) from the ground truth.

    Returns:
    - ate (float): Root Mean Square Error (RMSE) of the trajectory error.
    """
    if len(estimated_trajectory) != len(ground_truth_trajectory):
        raise ValueError("Estimated and ground truth trajectories must have the same length.")

    estimated_trajectory = np.array(estimated_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory)

    errors = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, axis=1)
    ate = np.sqrt(np.mean(errors**2))
    return ate


def compute_relative_pose_error(estimated_trajectory, ground_truth_trajectory):
    """
    Compute the Relative Pose Error (RPE) between the estimated trajectory and the ground truth.

    Parameters:
    - estimated_trajectory (list): List of 3D positions (Nx3) from the VO pipeline.
    - ground_truth_trajectory (list): List of 3D positions (Nx3) from the ground truth.

    Returns:
    - rpe (float): Mean Relative Pose Error.
    """
    if len(estimated_trajectory) < 2 or len(ground_truth_trajectory) < 2:
        raise ValueError("Trajectories must have at least two points to compute RPE.")

    estimated_trajectory = np.array(estimated_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory)

    relative_errors = []
    for i in range(1, len(estimated_trajectory)):
        est_delta = estimated_trajectory[i] - estimated_trajectory[i - 1]
        gt_delta = ground_truth_trajectory[i] - ground_truth_trajectory[i - 1]
        relative_error = np.linalg.norm(est_delta - gt_delta)
        relative_errors.append(relative_error)

    rpe = np.mean(relative_errors)
    return rpe


def compute_scale_drift(estimated_trajectory, ground_truth_trajectory):
    """
    Compute the scale drift between the estimated trajectory and the ground truth.

    Parameters:
    - estimated_trajectory (list): List of 3D positions (Nx3) from the VO pipeline.
    - ground_truth_trajectory (list): List of 3D positions (Nx3) from the ground truth.

    Returns:
    - scale_drift (float): Ratio of the estimated trajectory length to the ground truth trajectory length.
    """
    estimated_trajectory = np.array(estimated_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory)

    est_length = np.sum(np.linalg.norm(np.diff(estimated_trajectory, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(ground_truth_trajectory, axis=0), axis=1))

    if gt_length == 0:
        raise ValueError("Ground truth trajectory length is zero. Cannot compute scale drift.")

    scale_drift = est_length / gt_length
    return scale_drift


def compute_reprojection_error(points_2d, points_3d, camera_matrix):
    """
    Compute the reprojection error for 3D points projected onto 2D image space.

    Parameters:
    - points_2d (np.ndarray): Observed 2D points (Nx2).
    - points_3d (np.ndarray): Corresponding 3D points (Nx3).
    - camera_matrix (np.ndarray): Camera intrinsic matrix (3x3).

    Returns:
    - error (float): Mean reprojection error.
    """
    if points_2d.shape[1] != 2 or points_3d.shape[1] != 3:
        raise ValueError("2D points must have shape Nx2 and 3D points must have shape Nx3.")
    if camera_matrix.shape != (3, 3):
        raise ValueError("Camera matrix must have shape 3x3.")

    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Convert to homogeneous
    projected_points = camera_matrix @ points_3d_h.T
    projected_points /= projected_points[2]  # Normalize by depth
    projected_points = projected_points[:2].T

    error = np.linalg.norm(points_2d - projected_points, axis=1).mean()
    return error