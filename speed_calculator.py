
import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO
from scripts.tools.tracker import Tracker

# --- Configuration ---
VIDEO_PATH = 'sample.mp4'
MODEL_PATH = 'yolov8n.pt'
OUTPUT_PATH = 'output.avi'
SAVE_FRAMES = True
DISPLAY = True
LINE_DISTANCE = 10  # in meters
FRAME_SIZE = (1020, 500)
RED_LINE_Y = 198
BLUE_LINE_Y = 268
OFFSET = 6

def setup_output_folder(folder='detected_frames'):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_yolo_model(model_path):
    return YOLO(model_path)

def preprocess_frame(frame, size):
    return cv2.resize(frame, size)

def get_car_detections(results, class_list):
    boxes = results[0].boxes.data.detach().cpu().numpy()
    df = pd.DataFrame(boxes).astype("float")
    detections = []

    for _, row in df.iterrows():
        x1, y1, x2, y2, _, cls_id = map(int, row[:6])
        label = class_list[cls_id]
        if 'car' in label:
            detections.append([x1, y1, x2, y2])
    return detections

def calculate_speed(start_time, distance_m=10):
    elapsed_time = time.time() - start_time
    speed_ms = distance_m / elapsed_time
    return speed_ms * 3.6  # Convert m/s to km/h

def draw_lines_and_labels(frame, count_up, count_down):
    # Line annotations
    cv2.rectangle(frame, (0, 0), (250, 90), (0, 255, 255), -1)
    cv2.line(frame, (172, RED_LINE_Y), (774, RED_LINE_Y), (0, 0, 255), 2)
    cv2.putText(frame, 'Red Line', (172, RED_LINE_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.line(frame, (8, BLUE_LINE_Y), (927, BLUE_LINE_Y), (255, 0, 0), 2)
    cv2.putText(frame, 'Blue Line', (8, BLUE_LINE_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.putText(frame, f"Going Down - {count_down}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, f"Going Up - {count_up}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

def process_video():
    setup_output_folder()
    model = load_yolo_model(MODEL_PATH)
    class_list = model.model.names
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, FRAME_SIZE)

    count = 0
    down, up = {}, {}
    counter_down, counter_up = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        frame = preprocess_frame(frame, FRAME_SIZE)

        results = model.predict(frame)
        detections = get_car_detections(results, class_list)
        tracked_objects = tracker.update(detections)

        for x1, y1, x2, y2, obj_id in tracked_objects:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Downward movement
            if RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
                down[obj_id] = time.time()
            if obj_id in down and BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
                if obj_id not in counter_down:
                    speed = calculate_speed(down[obj_id], LINE_DISTANCE)
                    counter_down.append(obj_id)
                    annotate_vehicle(frame, x1, y1, x2, y2, cx, cy, obj_id, speed)

            # Upward movement
            if BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
                up[obj_id] = time.time()
            if obj_id in up and RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
                if obj_id not in counter_up:
                    speed = calculate_speed(up[obj_id], LINE_DISTANCE)
                    counter_up.append(obj_id)
                    annotate_vehicle(frame, x1, y1, x2, y2, cx, cy, obj_id, speed)

        draw_lines_and_labels(frame, len(counter_up), len(counter_down))

        if SAVE_FRAMES:
            cv2.imwrite(f'detected_frames/frame_{count}.jpg', frame)

        out.write(frame)

        if DISPLAY:
            cv2.imshow("Speed Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def annotate_vehicle(frame, x1, y1, x2, y2, cx, cy, obj_id, speed_kmh):
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, str(obj_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

# --- Entry Point ---
if __name__ == "__main__":
    process_video()