# Crab Tracker

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![Ultralytics](https://img.shields.io/badge/ultralytics-8.2.5-orange)

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
    model:
      path: # models weights
      conf_threshold: 0.8 # detection threshold
      nms_threshold: 0.4 # non maximal threshold (iou threshold in ultralisk)
      device: cpu
    input:
      path: # videos input
      extension: # videos extension
    output:
      save_results: true
      path: results/ # where to save the results this is the stats and the videos

## Example
    python main.py --config_path=config/run_conf.yaml
