from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from .tools import (get_car_id_given_a_plate, 
                   read_license_plate, 
                   write_csv,
)
from .speed_calculator import calculate_speed




def vehicle_and_plate_tracking(
    video_path: str,
    output_csv_path: str,
    coco_model_path: str = 'yolov8s.pt',
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
    video_frames = cv2.VideoCapture(video_path)

    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = video_frames.read()
        if ret:
            results[frame_nmr] = {}

            # Detect vehicles
            detections = coco_model(frame)[0]
            
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicle_classes:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))  # format: [x1, y1, x2, y2, score, track_id]

            # First, save all vehicles in the frame
            for track_id_data in track_ids:
                x1, y1, x2, y2, track_id = track_id_data[:5]  # track_id is the last element
                car_id = int(track_id)
                
                # Compute center of the car bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Initialize vehicle entry with default values
                results[frame_nmr][car_id] = {
                    'car': {
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy]
                    },
                    'license_plate': {
                        'bbox': None,
                        'text': None,
                        'bbox_score': None,
                        'text_score': None
                    }
                }

            # Detect license plates and update corresponding vehicles
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car_id_given_a_plate(license_plate, track_ids)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    # Update the vehicle's license plate information
                    if frame_nmr in results and car_id in results[frame_nmr]:
                        results[frame_nmr][car_id]['license_plate'] = {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }

    
    frame_height = int(video_frames.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results = calculate_speed(results, frame_height)
    video_frames.release()
    # Write results
    write_csv(results, output_csv_path)