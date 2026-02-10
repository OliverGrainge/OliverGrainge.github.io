---
title: "VSLAM: Stereo Visual SLAM Pipeline"
excerpt: "Pure-Python stereo SLAM pipeline (KITTI-compatible) with feature tracking, stereo matching, PnP/ICP motion estimation, and bundle adjustment."
collection: portfolio
header:
  teaser: vslam-image.jpg
tags:
  - Open Source
  - Computer Vision
  - Robotics
---

Visual Simultaneous Localization and Mapping (VSLAM) is a fundamental capability for autonomous robots and vehicles. This project implements a complete stereo visual SLAM pipeline in pure Python, designed for clarity and extensibility while remaining compatible with standard benchmarks.

[View the VSLAM GitHub Repository](https://github.com/OliverGrainge/VSLAM)

## Overview

The pipeline takes stereo image pairs and produces a 3D trajectory estimate of the camera's path through the environment. It combines classical computer vision techniques into a cohesive system that can be used for research, education, and prototyping.

## Pipeline Components

1. **Feature Detection & Tracking** -- Detects keypoints across stereo frames and tracks them temporally using optical flow, providing robust correspondences for motion estimation.

2. **Stereo Matching** -- Matches features between left and right camera images to triangulate 3D landmark positions using known camera geometry.

3. **Motion Estimation** -- Estimates camera pose changes between frames using PnP (Perspective-n-Point) for 2D-3D correspondences and ICP (Iterative Closest Point) for 3D-3D alignment.

4. **Bundle Adjustment** -- Jointly optimizes camera poses and 3D landmark positions over a sliding window to minimize reprojection error and improve trajectory consistency.

## Technical Stack

- **Python** with NumPy for core computation
- **OpenCV** for feature detection, matching, and image processing
- **KITTI benchmark** compatible input/output formats
- **pytest** test suite for validation of individual components

## Key Design Decisions

The pipeline is implemented entirely in Python to maximize readability and ease of modification. While this trades off raw performance compared to C++ implementations, it makes the codebase accessible for researchers wanting to experiment with individual components or students learning about SLAM systems.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- SciPy (for optimization)

## License

MIT License
