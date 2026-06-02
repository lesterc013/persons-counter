# Task 1 — Automated Person Counting

## Setup Instructions

- venv version: Python 3.14.3.
- Create venv and activate.
- Install requirements: Note: The installed PyTorch dependency has CUDA support for NVIDIA GPU usage.
- Place your input video file (TestVidTask.mov) in the root directory of the project.
- The YOLO model weight file (yolov8l.pt) will automatically download upon the first execution if it is not already present in the working directory.

## Core Dependencies

The core machine learning and computer vision engines used in this workflow are:

- Object Detection & Tracking Engine: ultralytics version 8.4.56 (utilizing the YOLOv8 Large pre-trained model: yolov8l.pt).

- Deep Learning Framework: torch version 2.11.0+cu128 with CUDA 12.8 support.

- Computer Vision Processing: opencv-python version 4.13.0.92.

## Run Instructions

- Run python counter.py in the CLI.
- A pop up window should show the object detection and counting.
- Window closes at the end of the file.
- Video is saved in object_counting_output.avi and the counts are logged to a txt file.

## Approach and Results

`counter.py` automates the detection and counting of persons crossing a defined gate line in the input video. The final count is used to derive a baseline arrival rate for the immigration checkpoint simulation.

Results are logged to `counts.txt` on each run.

### Approach

A YOLOv8 object detection model is used for person detection on each frame. A vertical gate line is drawn at a configurable position across the frame. When a tracked person's centroid crosses this line, the count increments.

The bottom half of the video (a glass reflection) was cropped out to reduce noise.

### Known Limitation — Top-Down Detection

The YOLOv8 model, and most of the other Object Detection ones, are pretrained on the COCO dataset, where "person" labels are predominantly captured at eye-level. Performance degrades on top-down footage as the model's learned features (face, shoulders, upright silhouette) do not match the aerial perspective.

Two possible mitigations were considered:

- **Fine-tune on top-down pedestrian data** (e.g. using VisDrone dataset). Would make a reliable and reusable model for unseen footage, but time-consuming to implement properly.
- **Confidence threshold sweep against a manual ground truth**. Running the base model across a list of confidence thresholds and then picking the one that is closest to a known ground truth. Quick to implement, but requires existing labelled data and may not generalise/scale for unseen data.

### Result

Given time constraints, I opted to demonstrate a working AI inference pipeline rather than pursue model fine-tuning. The model was run at several confidence thresholds and the range of count was around **18 to 20 persons over 23 seconds**.
