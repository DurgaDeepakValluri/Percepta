import numpy as np
import logging


def update_trajectory(trajectory_list: list, translation_vector: np.ndarray) -> list:
    """
    Update the trajectory with a new translation vector.

    Parameters:
    - trajectory_list (list): List of 3D trajectory points.
    - translation_vector (np.ndarray): 3D translation vector.

    Returns:
    - updated_trajectory (list): Updated trajectory list.

    Raises:
    - ValueError: If inputs are invalid.
    """
    if not isinstance(trajectory_list, list):
        raise ValueError("Trajectory must be a list of 3D points.")
    if not isinstance(translation_vector, np.ndarray) or translation_vector.shape != (3,):
        raise ValueError("Translation vector must be a numpy array of shape (3,).")

    if len(trajectory_list) == 0:
        trajectory_list.append(translation_vector.tolist())
    else:
        last_position = np.array(trajectory_list[-1])
        new_position = last_position + translation_vector
        trajectory_list.append(new_position.tolist())

    return trajectory_list


def save_trajectory(trajectory_list: list, save_path: str):
    """
    Save the trajectory to a file.

    Parameters:
    - trajectory_list (list): List of 3D trajectory points.
    - save_path (str): Path to save the trajectory file.

    Raises:
    - ValueError: If inputs are invalid.
    """
    if not isinstance(trajectory_list, list):
        raise ValueError("Trajectory must be a list of 3D points.")
    if not all(isinstance(point, (list, np.ndarray)) and len(point) == 3 for point in trajectory_list):
        raise ValueError("Each trajectory point must be a 3D coordinate.")

    try:
        np.savetxt(save_path, np.array(trajectory_list), fmt="%.6f", delimiter=",")
        logging.info(f"Trajectory saved to {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save trajectory to {save_path}: {e}")