import cv2
import numpy as np


def compute_essential_matrix(points1: np.ndarray, points2: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Essential Matrix from matched points.

    Parameters:
    - points1 (np.ndarray): Matched points in the first image (Nx2).
    - points2 (np.ndarray): Matched points in the second image (Nx2).
    - camera_matrix (np.ndarray): Camera intrinsic matrix (3x3).

    Returns:
    - E (np.ndarray): Essential Matrix (3x3).

    Raises:
    - ValueError: If inputs are invalid.
    """
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Input points must have shape Nx2.")
    if camera_matrix.shape != (3, 3):
        raise ValueError("Camera matrix must have shape 3x3.")

    E, _ = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise ValueError("Failed to compute Essential Matrix.")
    return E


def recover_pose(E: np.ndarray, points1: np.ndarray, points2: np.ndarray, camera_matrix: np.ndarray) -> tuple:
    """
    Recover the relative pose (R, t) from the Essential Matrix.

    Parameters:
    - E (np.ndarray): Essential Matrix (3x3).
    - points1 (np.ndarray): Matched points in the first image (Nx2).
    - points2 (np.ndarray): Matched points in the second image (Nx2).
    - camera_matrix (np.ndarray): Camera intrinsic matrix (3x3).

    Returns:
    - R (np.ndarray): Rotation matrix (3x3).
    - t (np.ndarray): Translation vector (3x1).

    Raises:
    - ValueError: If inputs are invalid.
    """
    if E.shape != (3, 3):
        raise ValueError("Essential Matrix must have shape 3x3.")
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Input points must have shape Nx2.")
    if camera_matrix.shape != (3, 3):
        raise ValueError("Camera matrix must have shape 3x3.")

    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
    return R, t


def triangulate_points(points1: np.ndarray, points2: np.ndarray, proj_matrix1: np.ndarray, proj_matrix2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from two views.

    Parameters:
    - points1 (np.ndarray): Matched points in the first image (Nx2).
    - points2 (np.ndarray): Matched points in the second image (Nx2).
    - proj_matrix1 (np.ndarray): Projection matrix for the first camera (3x4).
    - proj_matrix2 (np.ndarray): Projection matrix for the second camera (3x4).

    Returns:
    - points_3d (np.ndarray): Triangulated 3D points (Nx3).

    Raises:
    - ValueError: If inputs are invalid.
    """
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Input points must have shape Nx2.")
    if proj_matrix1.shape != (3, 4) or proj_matrix2.shape != (3, 4):
        raise ValueError("Projection matrices must have shape 3x4.")

    points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T  # Convert from homogeneous to 3D
    return points_3d


def compute_reprojection_error(points_2d: np.ndarray, points_3d: np.ndarray, proj_matrix: np.ndarray) -> float:
    """
    Compute the reprojection error for 3D points projected onto 2D image space.

    Parameters:
    - points_2d (np.ndarray): Observed 2D points (Nx2).
    - points_3d (np.ndarray): Corresponding 3D points (Nx3).
    - proj_matrix (np.ndarray): Projection matrix (3x4).

    Returns:
    - error (float): Mean reprojection error.
    """
    if points_2d.shape[1] != 2 or points_3d.shape[1] != 3:
        raise ValueError("2D points must have shape Nx2 and 3D points must have shape Nx3.")
    if proj_matrix.shape != (3, 4):
        raise ValueError("Projection matrix must have shape 3x4.")

    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Convert to homogeneous
    projected_points = proj_matrix @ points_3d_h.T
    projected_points /= projected_points[2]  # Normalize by depth
    projected_points = projected_points[:2].T

    error = np.linalg.norm(points_2d - projected_points, axis=1).mean()
    return error