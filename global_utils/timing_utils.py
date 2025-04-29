import time


def start_timer() -> float:
    """
    Start a timer.

    Returns:
    - start_time (float): Current time in seconds.
    """
    return time.time()


def end_timer(start_time: float) -> float:
    """
    End a timer and calculate elapsed time.

    Parameters:
    - start_time (float): Start time in seconds.

    Returns:
    - elapsed_time (float): Elapsed time in seconds.
    """
    return time.time() - start_time


def compute_fps(start_time: float, num_frames: int) -> float:
    """
    Compute frames per second (FPS).

    Parameters:
    - start_time (float): Start time in seconds.
    - num_frames (int): Number of frames processed.

    Returns:
    - fps (float): Frames per second.
    """
    elapsed_time = end_timer(start_time)
    return num_frames / elapsed_time if elapsed_time > 0 else 0.0