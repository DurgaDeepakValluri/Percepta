import os
import cv2
import glob
import requests
import numpy as np
from typing import Callable, List, Generator, Optional
import logging


def load_frames(source: str, preprocess_func: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> List:
    """
    Load frames from a video file, folder of images, webcam stream, URL, or a single image.

    Parameters:
    - source (str): Path to a video file, folder of images, stream source, URL, or single image.
    - preprocess_func (Callable): Function to preprocess each frame (e.g., resizing, grayscale conversion).
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - list_of_frames (List): List of preprocessed frames (for video, folder, or image URL inputs).

    Raises:
    - ValueError: If the source is invalid or unsupported.
    """
    if logger is None:
        logger = logging.getLogger("InputUtils")
        logging.basicConfig(level=logging.INFO)

    if os.path.isdir(source):
        return read_image_folder(source, preprocess_func, logger)
    elif os.path.isfile(source):
        if source.endswith((".jpg", ".jpeg", ".png")):
            return [read_single_image(source, preprocess_func, logger)]
        return read_video(source, preprocess_func, logger)
    elif source.startswith("http://") or source.startswith("https://"):
        if source.endswith((".jpg", ".jpeg", ".png")):
            return [read_image_from_url(source, preprocess_func, logger)]
        else:
            return list(read_stream(source, preprocess_func, logger))
    elif isinstance(source, (int, str)):
        return list(read_stream(source, preprocess_func, logger))
    else:
        raise ValueError("Invalid source. Must be a directory, video file, stream source, URL, or single image.")


def read_single_image(image_path: str, preprocess_func: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Read a single image from a file.

    Parameters:
    - image_path (str): Path to the image file.
    - preprocess_func (Callable): Function to preprocess the image.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - frame (np.ndarray): Preprocessed image.

    Raises:
    - ValueError: If the image cannot be read.
    """
    if logger is None:
        logger = logging.getLogger("InputUtils")

    if not os.path.exists(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if preprocess_func:
        frame = preprocess_func(frame)

    logger.info(f"Successfully loaded image: {image_path}")
    return frame


def read_image_from_url(url: str, preprocess_func: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Read an image from a URL.

    Parameters:
    - url (str): URL to the image.
    - preprocess_func (Callable): Function to preprocess the image.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - frame (np.ndarray): Preprocessed image.

    Raises:
    - ValueError: If the image cannot be downloaded or read.
    """
    if logger is None:
        logger = logging.getLogger("InputUtils")

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to decode image from URL: {url}")
        if preprocess_func:
            frame = preprocess_func(frame)
        logger.info(f"Successfully loaded image from URL: {url}")
        return frame
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading image from URL: {url}. Error: {e}")
        raise ValueError(f"Failed to load image from URL: {url}")


def read_video(video_path: str, preprocess_func: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> List:
    """
    Read frames from a video file.

    Parameters:
    - video_path (str): Path to the video file.
    - preprocess_func (Callable): Function to preprocess each frame.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - list_of_frames (List): List of preprocessed frames.

    Raises:
    - ValueError: If the video file cannot be opened.
    """
    if logger is None:
        logger = logging.getLogger("InputUtils")

    if not os.path.exists(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if preprocess_func:
            frame = preprocess_func(frame)
        frames.append(frame)

    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video: {video_path}")
    return frames


def read_image_folder(folder_path: str, preprocess_func: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> List:
    """
    Read frames from a folder of images.

    Parameters:
    - folder_path (str): Path to the folder containing images.
    - preprocess_func (Callable): Function to preprocess each frame.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - list_of_frames (List): List of preprocessed frames.

    Raises:
    - ValueError: If the folder does not exist or contains no images.
    """
    if logger is None:
        logger = logging.getLogger("InputUtils")

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
    if not image_paths:
        raise ValueError(f"No images found in folder: {folder_path}")

    frames = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            logger.warning(f"Failed to read image: {path}")
            continue
        if preprocess_func:
            frame = preprocess_func(frame)
        frames.append(frame)

    logger.info(f"Loaded {len(frames)} frames from folder: {folder_path}")
    return frames


def read_stream(stream_source=0, preprocess_func: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> Generator:
    """
    Read frames from a webcam, IP camera, or video stream URL.

    Parameters:
    - stream_source (int or str): Webcam index (e.g., 0) or stream URL.
    - preprocess_func (Callable): Function to preprocess each frame.
    - logger (logging.Logger): Logger for logging messages.

    Yields:
    - frame (ndarray): Preprocessed frame.

    Raises:
    - ValueError: If the stream cannot be opened.
    """
    if logger is None:
        logger = logging.getLogger("InputUtils")

    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open stream: {stream_source}")

    logger.info(f"Streaming frames from source: {stream_source}")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Stream ended or failed.")
            break
        if preprocess_func:
            frame = preprocess_func(frame)
        yield frame

    cap.release()
    logger.info("Stream closed.")