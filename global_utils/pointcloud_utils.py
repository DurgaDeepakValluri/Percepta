import open3d as o3d
import numpy as np
import logging


def update_pointcloud(pointcloud: o3d.geometry.PointCloud, new_points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Update the point cloud with new 3D points.

    Parameters:
    - pointcloud (o3d.geometry.PointCloud): Existing Open3D point cloud.
    - new_points (np.ndarray): New 3D points to add (shape: Nx3).

    Returns:
    - updated_pointcloud (o3d.geometry.PointCloud): Updated point cloud.

    Raises:
    - ValueError: If inputs are invalid.
    """
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise ValueError("Point cloud must be an Open3D PointCloud object.")
    if not isinstance(new_points, np.ndarray) or new_points.shape[1] != 3:
        raise ValueError("New points must be a numpy array of shape Nx3.")

    pointcloud.points.extend(o3d.utility.Vector3dVector(new_points))
    return pointcloud


def save_pointcloud(pointcloud: o3d.geometry.PointCloud, save_path: str):
    """
    Save the point cloud to a file.

    Parameters:
    - pointcloud (o3d.geometry.PointCloud): Open3D point cloud to save.
    - save_path (str): Path to save the point cloud file.

    Raises:
    - ValueError: If inputs are invalid.
    """
    if not isinstance(pointcloud, o3d.geometry.PointCloud):
        raise ValueError("Point cloud must be an Open3D PointCloud object.")

    try:
        o3d.io.write_point_cloud(save_path, pointcloud)
        logging.info(f"Point cloud saved to {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save point cloud to {save_path}: {e}")