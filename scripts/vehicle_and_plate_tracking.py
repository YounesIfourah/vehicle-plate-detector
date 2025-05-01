from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from .tools import (get_car, 
                   read_license_plate, 
                   write_csv
)

def vehicle_and_plate_tracking(
    video_path: str,
    output_csv_path: str,
    coco_model_path: str = 'yolov8n.pt',
    license_plate_model_path: str = 'license_plate_detector.pt',
    vehicle_classes: list = [2, 3, 5, 7]
):
    """
    Processes a video to detect vehicles and license plates, then writes results to a CSV file.
    
    Args:
        video_path (str): Path to input video.
        output_csv_path (str): Path to save the output CSV file.
        coco_model_path (str): Path to the YOLO model for detecting vehicles.
        license_plate_model_path (str): Path to the YOLO model for detecting license plates.
        vehicle_classes (list): List of COCO class IDs considered as vehicles.
    """
    results = {}
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_plate_model_path)

    # Load video
    cap = cv2.VideoCapture(video_path)

    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}

            # Detect vehicles
            # we return just the first element becasue we passed one frame
            detections = coco_model(frame)[0]

            # here is what a result may looks like :
            # Attributes:
            # boxes: tensor([[100.0, 200.0, 300.0, 400.0, 0.95, 2],
            #                [500.0, 600.0, 700.0, 800.0, 0.88, 3]])
            # masks: None
            # probs: None
            # orig_img: array shape (720, 1280, 3)
            # names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', ...}

            
            #In PyTorch (and YOLO uses PyTorch), a tensor is simply a multi-dimensional array optimized for fast calculations, especially on GPU.
            #In our case the tensor is a 2D matrix
            
            detections_ = []
            for detection in detections.boxes.data.tolist():   #.tolist() converts the tensor into a Python list, which makes it easier to loop through and work with inside Python code.
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicle_classes:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))  # add an id to look like [100, 150, 200, 250, 0.95, this is the id --> 1],

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

    cap.release()
    # Write results
    write_csv(results, output_csv_path)


