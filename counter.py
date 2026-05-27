import cv2
from ultralytics import solutions

# --- Config ---
VIDEO_PATH = "TestVidTask.mov"
OUTPUT_PATH = "object_counting_output.avi"
MODEL = "yolov8l.pt"
CONF_THRESHOLD = 0.05
GATE_POSITION = 0.5  # fraction across the frame (0.5 = midpoint)

# --- VideoCapture and Writer setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

video_writer = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h // 2)
)

# Set vertical gate line x-position at GATE_POSITION
gate_x = int(w * GATE_POSITION)
region_points = [(gate_x, 0), (gate_x, h)]

# --- Counter Object ---
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model=MODEL,
    classes=[
        0
    ],  # 0 = person (based on COCO dataset as explained in yolo documentation)
    conf=CONF_THRESHOLD,
)

# --- Start the counting process ---
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    # crop data to use to be the top half of the video only - the raw video's bottom half is reflection off glass and is considered noise.
    im0 = im0[: h // 2 :]
    results = counter(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# --- Results ---
# Note: ObjectCounter counts R to L as "OUT" as seen in the video; "IN" is the opposite.
# So, only count the R to L per the instructions provided.
print(f"Persons going R to L count: {counter.out_count}")
with open("counts.txt", "a") as f:
    f.write(f"Video: {VIDEO_PATH}\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Confidence threshold: {CONF_THRESHOLD}\n")
    f.write(f"Gate position: {GATE_POSITION} ({gate_x}px)\n")
    f.write(f"Persons going R to L count: {counter.out_count}\n")
    f.write("-" * 30 + "\n")
