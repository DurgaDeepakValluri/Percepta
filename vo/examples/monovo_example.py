import numpy as np
from percepta.vo import MonoVO

def main():
    # Define the camera intrinsic matrix
    camera_matrix = np.array([[718.856, 0, 607.1928],
                               [0, 718.856, 185.2157],
                               [0, 0, 1]])

    # Initialize the MonoVO pipeline
    vo = MonoVO(camera_matrix=camera_matrix)

    # Run the pipeline on a folder of images
    trajectory, pointcloud = vo.run_pipeline("path/to/image/folder")

    # Print the estimated trajectory
    print("Estimated Trajectory:", trajectory)

    # Save outputs (optional)
    vo.save_outputs(output_dir="outputs")

if __name__ == "__main__":
    main()