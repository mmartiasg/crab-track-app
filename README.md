# Crab Tracker

This Python application uses YOLO (You Only Look Once) from the `ultralytics` library to detect and track crabs in video feeds or images. YOLO is a real-time object detection system, and with the latest advancements in `ultralytics`, we can achieve efficient and accurate detection of crabs for research, environmental monitoring, or other applications.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Example](#example)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Real-time crab detection using YOLO from the `ultralytics` library.
- Configurable for different image and video sources.
- Options to save detection results as images or videos.
- Adjustable parameters to improve detection accuracy.
- Missing points are interpolated using the previous and next points.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/crab-tracker.git
   cd crab-tracker

## Configuration
    ```
        {
          "MODEL_WEIGHTS": "models/0.2.0/tracking.onnx",
          "VIDEO_FOLDER": "datasets/sample_dataset_icman",
          "RESULTS_PATH": "results",
          "VIDEO_EXTENSION": "mp4"
        }
    ```

## Example
    python track.py --config_path="config/run_conf.json"
