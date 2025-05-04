import cv2
import numpy as np
import pandas as pd
import ast
import os
from typing import Optional, Tuple, Dict

# Global configuration
SPEED_LIMIT = 130
BORDER_VALID_SPEED = (193, 255, 114)   # Light Green
BORDER_VIOLATION_SPEED = (0, 0, 255) # Red
BORDER_UNKNOWN_SPEED = (0, 255, 255) # Yellow
VIOLATIONS_FOLDER = "violations"

# Global speed display background colors
BG_NORMAL_SPEED = (104, 189, 12)      # Green  
BG_VIOLATION_SPEED = (0, 0, 255)           # Red
TEXT_COLOR = (255, 255, 255)               # White

def create_violation_folder(violation_id: int) -> str:
    """Create a folder for a speed violation with the given ID."""
    violation_dir = os.path.join(VIOLATIONS_FOLDER, str(violation_id))
    os.makedirs(violation_dir, exist_ok=True)
    return violation_dir

def save_violation_evidence(violation_dir: str, car_crop: np.ndarray, license_crop: np.ndarray, 
                           car_id: int, speed: float) -> None:
    """Save evidence images for a speed violation."""
    if car_crop is not None:
        cv2.imwrite(os.path.join(violation_dir, f"car_{car_id}.jpg"), car_crop)
    if license_crop is not None:
        cv2.imwrite(os.path.join(violation_dir, f"license_{car_id}.jpg"), license_crop)
    with open(os.path.join(violation_dir, "info.txt"), "w") as f:
        f.write(f"Car ID: {car_id}\nSpeed: {speed} km/h")

def draw_vehicle_info(frame: np.ndarray, 
                      top_left: Tuple[int, int], 
                      bottom_right: Tuple[int, int], 
                      speed: Optional[float] = None) -> np.ndarray:
    """
    Draw vehicle border and speed information by coordinating the two sub-functions.
    
    Args:
        frame: Input image frame
        top_left: (x1, y1) coordinates of top-left corner
        bottom_right: (x2, y2) coordinates of bottom-right corner
        speed: Optional speed value to display
        
    Returns:
        The modified frame with vehicle information drawn
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Determine border color based on speed
    if speed is None:
        border_color = BORDER_UNKNOWN_SPEED
    elif speed <= SPEED_LIMIT:
        border_color = BORDER_VALID_SPEED
    else:
        border_color = BORDER_VIOLATION_SPEED
    
    # First draw the border
    frame = draw_vehicle_border(frame, (x1, y1), (x2, y2), border_color)
    
    # Then draw speed if available
    if speed is not None:
        frame = draw_speed_info(frame, x1, y1, x2, y2, speed)
    
    return frame

def draw_vehicle_border(frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                        color: Tuple[int, int, int]) -> np.ndarray:
    """Draw vehicle border with dynamic thickness based on box width."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    box_width = x2 - x1
    thickness = max(4, min(8, int(box_width / 50)))  # thickness between 2 and 8 px

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def draw_speed_info(frame: np.ndarray, 
                    x1: float, y1: float, 
                    x2: float, y2: float, 
                    speed: float) -> np.ndarray:
    """Draw speed box centered above vehicle with tight background around text."""
    # Prepare text parameters first to determine box size
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{speed:.0f} km/h"
    font_scale = 0.7
    thickness = 2  # Increased from 1 to make text bolder
    
    # Get text dimensions
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate centered position above vehicle
    car_center_x = (x1 + x2) // 2
    text_x = car_center_x - (text_width // 2)
    
    # Position box slightly above vehicle (5px padding)
    box_top = int(y1 - text_height - baseline - 10)  # 10px above vehicle
    box_bottom = int(y1 - 10)  # 10px above vehicle
    
    # Add some padding around text (5px each side)
    padding = 5
    box_left = text_x - padding
    box_right = text_x + text_width + padding
    
    # Choose background color based on speed limit
    bg_color = BG_NORMAL_SPEED if speed <= SPEED_LIMIT else BG_VIOLATION_SPEED
    
    # Draw filled rectangle (background)
    cv2.rectangle(frame, 
                 (box_left, box_top), 
                 (box_right, box_bottom), 
                 bg_color, -1)
    
    # Calculate text position (centered in box)
    text_y = box_bottom - ((box_bottom - box_top - text_height) // 2) - baseline
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y),
               font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

    return frame

def get_best_license_plate_crop(results: pd.DataFrame, car_id: int, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Get the best license plate crop for a vehicle based on confidence score."""
    car_data = results[results['car_id'] == car_id]
    if len(car_data) == 0:
        return None
    
    # Handle cases where all scores are NaN
    if car_data['license_number_score'].isna().all():
        return None
        
    try:
        # Get the row with maximum score, ignoring NA values
        max_score_idx = car_data['license_number_score'].idxmax(skipna=True)
        best_row = car_data.loc[max_score_idx]
        
        if best_row['license_plate_bbox'] == 'None':
            return None

        # Save current frame position
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Set video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
        ret, frame = cap.read()
        if not ret:
            return None

        # Parse and validate bounding box
        bbox = parse_bbox_string(best_row['license_plate_bbox'])
        if bbox is None:
            # Restore frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            return None
            
        validated_bbox = validate_bbox(*bbox, frame.shape)
        if validated_bbox is None:
            # Restore frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            return None
            
        x1, y1, x2, y2 = validated_bbox
        crop = frame[y1:y2, x1:x2]
        
        # Restore frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        return crop
    except Exception as e:
        print(f"Error getting license plate crop for car {car_id}: {str(e)}")
        # Restore frame position in case of error
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        return None
    
def get_best_car_crop(results: pd.DataFrame, car_id: int, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Get a car crop for a vehicle (takes first valid crop if no score column exists)."""
    car_data = results[results['car_id'] == car_id]
    if len(car_data) == 0:
        return None
        
    # Save current frame position
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
    try:
        # If we have a car_score column, use the highest score crop
        if 'car_score' in car_data.columns:
            max_score_idx = car_data['car_score'].idxmax()
            best_row = car_data.loc[max_score_idx]
        else:
            # Otherwise just take the first row with a valid bbox
            valid_rows = car_data[car_data['car_bbox'].notna() & (car_data['car_bbox'] != 'None')]
            if len(valid_rows) == 0:
                # Restore frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                return None
            best_row = valid_rows.iloc[0]
        
        # Set video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
        ret, frame = cap.read()
        if not ret:
            # Restore frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            return None

        # Parse and validate bounding box
        bbox = parse_bbox_string(best_row['car_bbox'])
        if bbox is None:
            # Restore frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            return None
            
        validated_bbox = validate_bbox(*bbox, frame.shape)
        if validated_bbox is None:
            # Restore frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            return None
            
        x1, y1, x2, y2 = validated_bbox
        crop = frame[y1:y2, x1:x2]
        
        # Restore frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        return crop
    except Exception as e:
        print(f"Error getting car crop for car {car_id}: {str(e)}")
        # Restore frame position in case of error
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        return None

def process_video(video_path: str, results_csv_path: str, output_video_path: str) -> None:
    """Main function to process video and detect speed violations."""
    # Load results
    try:
        results = pd.read_csv(results_csv_path)
    except Exception as e:
        print(f"Error loading results CSV: {str(e)}")
        return

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare video writer with better settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

    # Create violations folder
    os.makedirs(VIOLATIONS_FOLDER, exist_ok=True)
    
    # Dictionary to track maximum speed per car
    car_max_speeds = {}
    # Dictionary to track violation folders per car
    car_violation_folders = {}

    # Process each frame sequentially
    frame_nmr = 0
    last_reported_frame = -1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Verify we're processing frames in order
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame_pos != frame_nmr + 1:
            # If we've jumped frames, seek to the correct position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr + 1)
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        frame_nmr = current_frame_pos
        
        if frame_nmr % 100 == 0 and frame_nmr != last_reported_frame:
            print(f"Processing frame {frame_nmr}/{total_frames}")
            last_reported_frame = frame_nmr

        try:
            df_frame = results[results['frame_nmr'] == frame_nmr]

            for _, row in df_frame.iterrows():
                try:
                    car_bbox = parse_bbox_string(row['car_bbox'])
                    if car_bbox is None:
                        continue
                        
                    car_x1, car_y1, car_x2, car_y2 = car_bbox
                    car_id = row['car_id']
                    speed = row.get('speed', None)
                    
                    try:
                        speed = float(speed) if speed is not None and not pd.isna(speed) else None
                    except (ValueError, TypeError):
                        speed = None
                    
                    # Validate bbox coordinates first
                    validated_bbox = validate_bbox(car_x1, car_y1, car_x2, car_y2, frame.shape)
                    if validated_bbox is None:
                        continue
                    
                    x1, y1, x2, y2 = validated_bbox
                    
                    # Determine border color based on speed
                    if speed is None:
                        border_color = BORDER_UNKNOWN_SPEED
                    elif speed > SPEED_LIMIT:
                        border_color = BORDER_VIOLATION_SPEED
                        
                        # Check if this is a new maximum speed for this car
                        if car_id not in car_max_speeds or speed > car_max_speeds[car_id]:
                            car_max_speeds[car_id] = speed
                            
                            # Get best crops for evidence
                            license_crop = get_best_license_plate_crop(results, car_id, cap)
                            car_crop = get_best_car_crop(results, car_id, cap)
                            
                            # Create or update violation folder
                            if car_id not in car_violation_folders:
                                violation_dir = create_violation_folder(car_id)
                                car_violation_folders[car_id] = violation_dir
                            else:
                                violation_dir = car_violation_folders[car_id]
                            
                            # Save evidence
                            save_violation_evidence(violation_dir, car_crop, license_crop, car_id, speed)
                    else:
                        border_color = BORDER_VALID_SPEED
                    
                    frame = draw_vehicle_info(frame, (x1, y1), (x2, y2), speed)
                        
                except Exception as e:
                    print(f"Error processing vehicle in frame {frame_nmr}: {str(e)}")
                    continue
                    
            out.write(frame)
        except Exception as e:
            print(f"Error processing frame {frame_nmr}: {str(e)}")
            continue

    # Clean up
    out.release()
    cap.release()
    print(f"Processing complete. Found {len(car_violation_folders)} speed violations.")
    print(f"Output saved to {output_video_path}")

def parse_bbox_string(bbox_str: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse bounding box string into coordinates with better error handling."""
    if pd.isna(bbox_str) or bbox_str == 'None' or not bbox_str.strip():
        return None
        
    try:
        # Clean the string and convert to list
        bbox_str = bbox_str.strip()
        if bbox_str.startswith('[') and bbox_str.endswith(']'):
            bbox_str = bbox_str[1:-1]
        
        # Handle different separator formats
        if ',' in bbox_str:
            parts = [x.strip() for x in bbox_str.split(',')]
        else:
            parts = [x.strip() for x in bbox_str.split()]
        
        if len(parts) != 4:
            return None
            
        return tuple(float(x) for x in parts)
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing bbox string '{bbox_str}': {str(e)}")
        return None

def validate_bbox(x1: float, y1: float, x2: float, y2: float, 
                 frame_shape: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    """Validate and adjust bounding box coordinates with better rounding."""
    try:
        # Round coordinates first
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        
        # Ensure valid box dimensions
        if x2 <= x1 or y2 <= y1:
            return None
            
        if frame_shape is not None:
            h, w = frame_shape[:2]
            # Clamp coordinates to frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2 <= x1 or y2 <= y1:
                return None
                
        return (x1, y1, x2, y2)
    except (ValueError, TypeError) as e:
        print(f"Error validating bbox [{x1}, {y1}, {x2}, {y2}]: {str(e)}")
        return None



