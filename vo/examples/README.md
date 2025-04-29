# Examples for Percepta VO Pipelines

This folder contains example scripts for running the Visual Odometry (VO) pipelines.

## Scripts

1. **MonoVO Example** (`monovo_example.py`):
   - Demonstrates how to use the MonoVO pipeline with a folder of monocular images.

2. **StereoVO Example** (`stereovo_example.py`):
   - Demonstrates how to use the StereoVO pipeline with a folder of stereo image pairs.

3. **RGBDVO Example** (`rgbdvo_example.py`):
   - Demonstrates how to use the RGBDVO pipeline with a folder of RGB-D image pairs.

## How to Run

1. Install the required dependencies:
```bash
   pip install -r requirements.txt
```

2. Run the desired example script
```bash
python examples/monovo_example.py
```

3. Outputs will be saved in the **outputs/** folder