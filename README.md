# Task 1 — Automated Person Counting

`counter.py` automates the detection and counting of persons crossing a defined gate line in the input video. The final count is used to derive a baseline arrival rate for the immigration checkpoint simulation.

Results are logged to `counts.txt` on each run.

## Approach

A YOLOv8 object detection model is used for person detection on each frame. A vertical gate line is drawn at a configurable position across the frame. When a tracked person's centroid crosses this line, the count increments.

The bottom half of the video (a glass reflection) was cropped out to reduce noise.

## Known Limitation — Top-Down Detection

The YOLOv8 model, and most of the other Object Detection ones, are pretrained on the COCO dataset, where "person" labels are predominantly captured at eye-level. Performance degrades on top-down footage as the model's learned features (face, shoulders, upright silhouette) do not match the aerial perspective.

Two possible mitigations were considered:

- **Fine-tune on top-down pedestrian data** (e.g. using VisDrone dataset). Would make a reliable and reusable model for unseen footage, but time-consuming to implement properly.
- **Confidence threshold sweep against a manual ground truth**. Running the base model across a list of confidence thresholds and then picking the one that is closest to a known ground truth. Quick to implement, but requires existing labelled data and may not generalise/scale for unseen data.

## Result

Given time constraints, I opted to demonstrate a working AI inference pipeline rather than pursue model fine-tuning. The model was run at several confidence thresholds and a count of **20 persons over 23 seconds** was selected as the most reasonable estimate against a personal manual count of 34.

This gives a conservative baseline arrival rate of **~1 person/second**, which will be used as in the simulation.
