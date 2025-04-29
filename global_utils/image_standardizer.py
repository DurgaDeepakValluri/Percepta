import cv2
import numpy as np
from typing import Optional, Tuple


def preprocess_frame(frame: np.ndarray, input_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Standardize a frame before processing.
    - Converts to grayscale if necessary.
    - Resizes to the target size if specified.
    - Ensures dtype is uint8.

    Parameters:
    - frame (np.ndarray): Input frame.
    - input_size (Optional[Tuple[int, int]]): Target size (width, height) for resizing.

    Returns:
    - standardized_frame (np.ndarray): Preprocessed frame.

    Raises:
    - ValueError: If the input frame is invalid.
    """
    validate_frame(frame)

    frame = convert_to_grayscale(frame)
    if input_size:
        frame = resize_frame(frame, input_size)
    frame = normalize_dtype(frame)

    return frame


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a frame to the specified target size.

    Parameters:
    - frame (np.ndarray): Input frame.
    - target_size (Tuple[int, int]): Target size (width, height).

    Returns:
    - resized_frame (np.ndarray): Resized frame.

    Raises:
    - ValueError: If the target size is invalid.
    """
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError("Target size must be a tuple of two integers (width, height).")
    if not all(isinstance(dim, int) and dim > 0 for dim in target_size):
        raise ValueError("Target size dimensions must be positive integers.")

    return cv2.resize(frame, target_size)


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert a frame to grayscale if it is not already.

    Parameters:
    - frame (np.ndarray): Input frame.

    Returns:
    - gray_frame (np.ndarray): Grayscale frame.
    """
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # Color image
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame  # Already grayscale


def normalize_dtype(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a frame's dtype to uint8.

    Parameters:
    - frame (np.ndarray): Input frame.

    Returns:
    - uint8_frame (np.ndarray): Frame with dtype uint8.

    Raises:
    - ValueError: If the frame contains invalid pixel values.
    """
    if frame.dtype == np.uint8:
        return frame  # Already uint8

    if frame.dtype in [np.float32, np.float64]:
        if np.max(frame) <= 1.0:  # Assume normalized float in range [0, 1]
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            raise ValueError("Float frames must have pixel values in the range [0, 1].")
    else:
        frame = frame.clip(0, 255).astype(np.uint8)  # Handle other integer types

    return frame


def validate_frame(frame: np.ndarray):
    """
    Validate the input frame to ensure it is a valid image.

    Parameters:
    - frame (np.ndarray): Input frame.

    Raises:
    - ValueError: If the frame is invalid.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a numpy ndarray.")
    if len(frame.shape) not in [2, 3]:
        raise ValueError("Input frame must be a 2D grayscale or 3D color image.")
    if len(frame.shape) == 3 and frame.shape[2] != 3:
        raise ValueError("3D frames must have exactly 3 channels (color image).")
    if frame.size == 0:
        raise ValueError("Input frame is empty.")