import cv2
import numpy as np


def track_features_klt(prev_frame: np.ndarray, curr_frame: np.ndarray, prev_pts: np.ndarray) -> tuple:
    """
    Track features between two frames using KLT (Kanade-Lucas-Tomasi) optical flow.

    Parameters:
    - prev_frame (np.ndarray): Previous grayscale frame.
    - curr_frame (np.ndarray): Current grayscale frame.
    - prev_pts (np.ndarray): Points to track in the previous frame (Nx2).

    Returns:
    - curr_pts (np.ndarray): Tracked points in the current frame (Nx2).
    - status (np.ndarray): Status array (1 if tracked successfully, 0 otherwise).
    - err (np.ndarray): Error array for each point.

    Raises:
    - ValueError: If inputs are invalid.
    """
    if prev_frame.ndim != 2 or curr_frame.ndim != 2:
        raise ValueError("Frames must be grayscale images.")
    if prev_pts.shape[1] != 2:
        raise ValueError("Previous points must have shape Nx2.")

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None, winSize=(21, 21), maxLevel=3)
    return curr_pts, status, err


def filter_tracked_features(prev_pts: np.ndarray, curr_pts: np.ndarray, status: np.ndarray) -> tuple:
    """
    Filter out points that were not successfully tracked.

    Parameters:
    - prev_pts (np.ndarray): Points in the previous frame (Nx2).
    - curr_pts (np.ndarray): Points in the current frame (Nx2).
    - status (np.ndarray): Status array (1 if tracked successfully, 0 otherwise).

    Returns:
    - filtered_prev_pts (np.ndarray): Successfully tracked points in the previous frame.
    - filtered_curr_pts (np.ndarray): Successfully tracked points in the current frame.
    """
    if prev_pts.shape[1] != 2 or curr_pts.shape[1] != 2:
        raise ValueError("Points must have shape Nx2.")
    if status.shape[0] != prev_pts.shape[0]:
        raise ValueError("Status array must have the same length as points.")

    mask = status.flatten() == 1
    return prev_pts[mask], curr_pts[mask]
