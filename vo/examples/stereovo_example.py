import numpy as np
from percepta.vo import StereoVO

def main():
    # Define the stereo camera intrinsic matrix
    camera_matrix = np.array([[718.856, 0, 607.1928],
                               [0, 718.856, 185.2157],
                               [0, 0, 1]])

    # Initialize the StereoVO pipeline
    vo = StereoVO(camera_matrix=camera_matrix)

    # Run the pipeline on a folder of stereo image pairs
    trajectory, pointcloud = vo.run_pipeline("path/to/stereo/image/folder")

    # Print the estimated trajectory
    print("Estimated Trajectory:", trajectory)

    # Save outputs (optional)
    vo.save_outputs(output_dir="outputs")

if __name__ == "__main__":
    main()