import cv2
from typing import Tuple, List, Optional


def initialize_detector(name: str):
    """
    Initialize a feature detector.

    Parameters:
    - name (str): Name of the detector ('ORB', 'SIFT', 'SURF').

    Returns:
    - detector: OpenCV feature detector object, or None if unavailable.

    Raises:
    - ValueError: If the detector name is invalid.
    """
    detectors = {
        'ORB': cv2.ORB_create,
        'SIFT': cv2.SIFT_create,
        'SURF': lambda: cv2.xfeatures2d.SURF_create() if hasattr(cv2, 'xfeatures2d') else None,
    }

    if name not in detectors:
        raise ValueError(f"Invalid detector name: {name}. Supported detectors are: {list(detectors.keys())}.")

    detector = detectors[name]()
    if detector is None:
        # Log a warning instead of raising an exception
        print(f"Warning: {name} detector is unavailable. Ensure OpenCV contrib modules are installed.")
    
    return detector


def initialize_matcher(detector_name: str, matcher_name: str):
    """
    Initialize a feature matcher.

    Parameters:
    - detector_name (str): Name of the detector ('ORB', 'SIFT', 'SURF').
    - matcher_name (str): Name of the matcher ('BF', 'FLANN').

    Returns:
    - matcher: OpenCV feature matcher object.

    Raises:
    - ValueError: If the matcher name is invalid.
    """
    if matcher_name == 'BF':
        norm = cv2.NORM_HAMMING if detector_name == 'ORB' else cv2.NORM_L2
        return cv2.BFMatcher(norm, crossCheck=True)
    elif matcher_name == 'FLANN':
        index_params = dict(algorithm=1, trees=5)  # FLANN KDTree index
        search_params = dict(checks=50)  # Number of checks
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Invalid matcher name: {matcher_name}. Supported matchers are: ['BF', 'FLANN'].")


def detect_and_compute(detector, frame) -> Tuple[List[cv2.KeyPoint], Optional[cv2.Mat]]:
    """
    Detect keypoints and compute descriptors for a given frame.

    Parameters:
    - detector: OpenCV feature detector object.
    - frame (ndarray): Input image/frame.

    Returns:
    - keypoints (list): List of detected keypoints.
    - descriptors (ndarray or None): Feature descriptors, or None if no descriptors are found.

    Raises:
    - ValueError: If the frame is invalid or None.
    """
    if frame is None:
        raise ValueError("Input frame is None. Please provide a valid image.")

    keypoints, descriptors = detector.detectAndCompute(frame, None)
    if not keypoints:
        # Gracefully handle cases where no keypoints are detected
        print("Warning: No keypoints detected in the frame.")
        return [], None

    return keypoints, descriptors