import numpy as np


def fuse_imu_with_vo(R: np.ndarray, t: np.ndarray, imu_data: dict) -> tuple:
    """
    Fuse IMU data with Visual Odometry (VO) pose estimates.

    Parameters:
    - R (np.ndarray): Rotation matrix from VO (3x3).
    - t (np.ndarray): Translation vector from VO (3x1).
    - imu_data (dict): IMU data containing 'gyro' (angular velocity) and 'accel' (linear acceleration).

    Returns:
    - fused_R (np.ndarray): Fused rotation matrix (3x3).
    - fused_t (np.ndarray): Fused translation vector (3x1).

    Raises:
    - ValueError: If inputs are invalid.
    """
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape 3x3.")
    if t.shape != (3, 1):
        raise ValueError("Translation vector must have shape 3x1.")
    if not isinstance(imu_data, dict) or 'gyro' not in imu_data or 'accel' not in imu_data:
        raise ValueError("IMU data must be a dictionary containing 'gyro' and 'accel'.")

    gyro_correction = np.eye(3) + 0.01 * np.array(imu_data['gyro'])  # Small correction from gyro
    fused_R = R @ gyro_correction
    fused_t = t + 0.01 * np.array(imu_data['accel']).reshape(3, 1)  # Small correction from acceleration

    return fused_R, fused_t


def integrate_imu(gyro: np.ndarray, accel: np.ndarray, dt: float) -> tuple:
    """
    Integrate IMU data to estimate rotation and translation.

    Parameters:
    - gyro (np.ndarray): Angular velocity (3x1).
    - accel (np.ndarray): Linear acceleration (3x1).
    - dt (float): Time step in seconds.

    Returns:
    - delta_R (np.ndarray): Estimated rotation matrix (3x3).
    - delta_t (np.ndarray): Estimated translation vector (3x1).
    """
    if gyro.shape != (3,) or accel.shape != (3,):
        raise ValueError("Gyro and accel must have shape (3,).")

    theta = np.linalg.norm(gyro) * dt
    if theta > 0:
        axis = gyro / np.linalg.norm(gyro)
        delta_R = cv2.Rodrigues(axis * theta)[0]
    else:
        delta_R = np.eye(3)

    delta_t = accel * dt**2 / 2
    return delta_R, delta_t