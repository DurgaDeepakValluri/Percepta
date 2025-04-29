import numpy as np


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.

    Parameters:
    - vec (np.ndarray): Input vector.

    Returns:
    - normalized_vec (np.ndarray): Normalized vector.

    Raises:
    - ValueError: If the vector has zero magnitude.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vec / norm


def skew_symmetric(vec: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a vector.

    Parameters:
    - vec (np.ndarray): Input vector (shape: 3,).

    Returns:
    - skew_matrix (np.ndarray): Skew-symmetric matrix (shape: 3x3).
    """
    if vec.shape != (3,):
        raise ValueError("Input vector must have shape (3,).")
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])


def safe_inverse(mat: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a matrix safely.

    Parameters:
    - mat (np.ndarray): Input matrix.

    Returns:
    - inverse (np.ndarray): Inverse of the matrix.

    Raises:
    - ValueError: If the matrix is singular.
    """
    if np.linalg.det(mat) == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return np.linalg.inv(mat)