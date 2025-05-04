import cv2
import pandas as pd
import numpy as np
import ast
from typing import Dict, Tuple, Optional



def draw_border(img: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 5, 
                line_length_x: int = 100, line_length_y: int = 100) -> np.ndarray:
    """Draw a border with corner lines around a bounding box.
    
    Args:
        img: Input image
        top_left: (x1, y1) coordinates of top-left corner
        bottom_right: (x2, y2) coordinates of bottom-right corner
        color: Border color (BGR format)
        thickness: Base thickness of border lines
        line_length_x: Length of horizontal border lines
        line_length_y: Length of vertical border lines
    
    Returns:
        Image with border drawn
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Calculate box size to determine thickness
    box_width = x2 - x1
    box_height = y2 - y1
    box_size = (box_width + box_height) / 2
    
    # Scale thickness based on box size
    scale_factor = min(1.0, box_size / 300)
    thickness = max(1, int(thickness * scale_factor))
    line_length_x = max(10, int(line_length_x * scale_factor))
    line_length_y = max(10, int(line_length_y * scale_factor))

    # Draw corner lines
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left vertical
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)  # top-left horizontal

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left vertical
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)  # bottom-left horizontal

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right horizontal
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)  # top-right vertical

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right vertical
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)  # bottom-right horizontal

    return img

def parse_bbox_string(bbox_str: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse bounding box string into coordinates.
    
    Args:
        bbox_str: String representation of bounding box (e.g., "[x1 y1 x2 y2]")
    
    Returns:
        Tuple of (x1, y1, x2, y2) or None if invalid
    """
    if pd.isna(bbox_str) or bbox_str == 'None' or not bbox_str.strip():
        return None
        
    try:
        # Clean and standardize the string format
        bbox_str = bbox_str.strip()
        bbox_str = bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        bbox = ast.literal_eval(bbox_str)
        
        if len(bbox) != 4:
            return None
            
        return tuple(map(float, bbox))
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing bbox string '{bbox_str}': {str(e)}")
        return None

def validate_bbox(x1: float, y1: float, x2: float, y2: float, 
                 frame_shape: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    """Validate and adjust bounding box coordinates.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        frame_shape: Optional (height, width) of the frame for boundary checking
    
    Returns:
        Validated and adjusted coordinates or None if invalid
    """
    try:
        # Convert to integers
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        
        # Basic validation
        if x2 <= x1 or y2 <= y1:
            return None
            
        if frame_shape is not None:
            h, w = frame_shape[:2]
            # Adjust coordinates to be within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # Check again after adjustment
            if x2 <= x1 or y2 <= y1:
                return None
                
        return (x1, y1, x2, y2)
    except (ValueError, TypeError) as e:
        print(f"Error validating bbox [{x1}, {y1}, {x2}, {y2}]: {str(e)}")
        return None

def prepare_license_plate_crops(results: pd.DataFrame, cap: cv2.VideoCapture) -> Dict[int, dict]:
    """Prepare license plate crops for all cars that have plates.
    
    Args:
        results: DataFrame containing detection results
        cap: VideoCapture object for the input video
    
    Returns:
        Dictionary mapping car_id to license plate data
    """
    license_plate_data = {}
    cars_with_plates = results[results['license_number'].notna()]
    
    for car_id in np.unique(cars_with_plates['car_id']):
        try:
            # Get the frame with highest score license plate for this car
            car_data = cars_with_plates[cars_with_plates['car_id'] == car_id]
            if len(car_data) == 0:
                continue
                
            max_score_idx = car_data['license_number_score'].idxmax()
            best_row = car_data.loc[max_score_idx]
            
            license_plate_data[car_id] = {
                'license_crop': None,
                'license_plate_number': best_row['license_number']
            }

            if best_row['license_plate_bbox'] == 'None':
                continue

            # Set video to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
            ret, frame = cap.read()
            
            if not ret:
                continue

            # Parse and validate bounding box
            bbox = parse_bbox_string(best_row['license_plate_bbox'])
            if bbox is None:
                continue
                
            validated_bbox = validate_bbox(*bbox, frame.shape)
            if validated_bbox is None:
                continue
                
            x1, y1, x2, y2 = validated_bbox

            # Extract and process license plate crop
            try:
                license_crop = frame[y1:y2, x1:x2]
                if license_crop.size == 0:
                    continue
                    
                # Calculate new dimensions maintaining aspect ratio
                crop_height = y2 - y1
                if crop_height == 0:
                    continue
                
                aspect_ratio = (x2 - x1) / crop_height
                new_width = int(400 * aspect_ratio)
                if new_width <= 0:
                    continue
                
                license_crop = cv2.resize(license_crop, (new_width, 400))
                license_plate_data[car_id]['license_crop'] = license_crop
            except Exception as e:
                print(f"Error processing license plate crop for car {car_id}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error processing car {car_id}: {str(e)}")
            continue

    return license_plate_data

def draw_car_annotation(frame: np.ndarray, car_x1: float, car_y1: float, 
                        car_x2: float, car_y2: float, has_plate: bool) -> Tuple[int, float]:
    """Draw car bounding box with appropriate style based on plate presence.
    
    Args:
        frame: Input frame
        car_x1, car_y1, car_x2, car_y2: Car bounding box coordinates
        has_plate: Whether the car has a detected license plate
    
    Returns:
        Tuple of (thickness, scale_factor) used for drawing
    """
    box_width = car_x2 - car_x1
    box_height = car_y2 - car_y1
    box_size = (box_width + box_height) / 2
    
    # Base thickness - thicker if plate detected
    base_thickness = 12 if has_plate else 5
    
    # Scale thickness based on box size
    scale_factor = min(1.0, box_size / 300)
    thickness = max(2, int(base_thickness * scale_factor))
    
    # Draw car bounding box with adjusted thickness
    draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), 
               (0, 255, 0) if has_plate else (0, 165, 255),  # Green if plate, orange if no plate
               thickness)
    
    return thickness, scale_factor

def overlay_license_plate(frame: np.ndarray, license_crop: Optional[np.ndarray], 
                          car_x1: float, car_y1: float, car_x2: float, car_y2: float, 
                          thickness: int, scale_factor: float) -> np.ndarray:
    """Overlay the license plate crop above the car bounding box.
    
    Args:
        frame: Input frame
        license_crop: License plate image crop
        car_x1, car_y1, car_x2, car_y2: Car bounding box coordinates
        thickness: Border thickness (used for scaling)
        scale_factor: Scaling factor based on car size
    
    Returns:
        Frame with license plate overlay
    """
    if license_crop is None:
        return frame

    H, W = license_crop.shape[:2]

    # Scale overlay size based on car distance
    overlay_scale = min(1.0, (car_x2 - car_x1) / 200) * scale_factor
    display_h = int(H * overlay_scale)
    display_w = int(W * overlay_scale)

    if display_h > 0 and display_w > 0:
        try:
            license_crop_scaled = cv2.resize(license_crop, (display_w, display_h))

            # Calculate position for the overlay
            y_start = int(car_y1) - display_h - int(50 * overlay_scale)
            y_end = int(car_y1) - int(50 * overlay_scale)
            x_center = int((car_x2 + car_x1) / 2)
            x_start = x_center - display_w // 2
            x_end = x_center + display_w // 2

            # Ensure coordinates are within frame bounds
            y_end = min(y_end, frame.shape[0])
            x_end = min(x_end, frame.shape[1])
            y_start = max(y_start, 0)
            x_start = max(x_start, 0)

            # Apply the overlay
            frame[y_start:y_end, x_start:x_end] = license_crop_scaled[:max(0, y_end - y_start), 
                                                                    :max(0, x_end - x_start)]
        except Exception as e:
            print(f"Error overlaying license plate: {str(e)}")

    return frame

def draw_speed_info(frame: np.ndarray, car_x1: float, car_y1: float, 
                    car_x2: float, car_y2: float, speed: float) -> np.ndarray:
    """Draw speed information above the vehicle bounding box."""
    if pd.isna(speed) or speed == 'None':
        return frame
    
    try:
        speed = float(speed)
    except (ValueError, TypeError):
        return frame
    
    # Calculate position for speed text (centered above bounding box)
    text = f"{speed:.1f} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Get text size first to properly position it
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text centered above the bounding box
    text_x = int((car_x1 + car_x2) / 2 - text_width / 2)
    text_y = int(car_y1) - 10  # 10 pixels above the top
    
    # Ensure text stays within frame bounds
    text_y = max(text_height + 5, text_y)  # At least text height +5px from top
    text_x = max(5, text_x)  # At least 5px from left
    text_x = min(frame.shape[1] - text_width - 5, text_x)  # At least 5px from right
    
    # Draw black background rectangle for better readability
    cv2.rectangle(frame, 
                 (text_x - 5, text_y - text_height - 5),
                 (text_x + text_width + 5, text_y + 5),
                 (0, 0, 0), -1)  # -1 means filled rectangle
    
    # Draw white speed text
    cv2.putText(frame, text, (text_x, text_y), 
               font, font_scale, (255, 255, 255), thickness)
    
    return frame

def process_frame(frame: np.ndarray, frame_nmr: int, results: pd.DataFrame, 
                 license_plate_data: Dict[int, dict]) -> np.ndarray:
    """Process a single frame and draw all annotations."""
    df_frame = results[results['frame_nmr'] == frame_nmr]

    for _, row in df_frame.iterrows():
        try:
            # Get car bounding box
            car_bbox = parse_bbox_string(row['car_bbox'])
            if car_bbox is None:
                continue
                
            car_x1, car_y1, car_x2, car_y2 = car_bbox
            car_id = row['car_id']
            
            # Determine if this car has a license plate
            has_plate = car_id in license_plate_data and row['license_plate_bbox'] != 'None'
            
            # Draw car annotation
            thickness, scale_factor = draw_car_annotation(frame, car_x1, car_y1, car_x2, car_y2, has_plate)
            
            # Draw speed information if available (check both 'speed' and 'speed_kph')
            speed = row.get('speed', None)

            # print(speed)

            if speed is not None and not pd.isna(speed):
                frame = draw_speed_info(frame, car_x1, car_y1, car_x2, car_y2, speed)
            
            # Process license plate if exists
            if has_plate:
                lp_bbox = parse_bbox_string(row['license_plate_bbox'])
                if lp_bbox is not None:
                    validated_bbox = validate_bbox(*lp_bbox, frame.shape)
                    if validated_bbox is not None:
                        x1, y1, x2, y2 = validated_bbox
                        # Draw license plate bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), max(3, thickness//2))

                        # Overlay license plate crop
                        frame = overlay_license_plate(
                            frame, 
                            license_plate_data[car_id]['license_crop'], 
                            car_x1, car_y1, car_x2, car_y2, 
                            thickness, scale_factor
                        )
        except Exception as e:
            print(f"Error processing car in frame {frame_nmr}: {str(e)}")
            continue
            
    return frame

def overlay_license_plate_on_video(video_path: str, results_csv_path: str, output_video_path: str) -> None:
    """Main function to process video and overlay information."""
    # Load results CSV with proper speed column handling
    try:
        results = pd.read_csv(results_csv_path)
        # Ensure speed column exists (check both variants)
        if 'speed_kph' not in results.columns and 'speed' in results.columns:
            results['speed_kph'] = results['speed']
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

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Prepare license plates data
    license_plate_data = prepare_license_plate_crops(results, cap)

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process each frame with progress feedback
    frame_nmr = -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        # Print progress every 100 frames
        if frame_nmr % 100 == 0:
            print(f"Processing frame {frame_nmr}/{total_frames}")

        try:
            frame = process_frame(frame, frame_nmr, results, license_plate_data)
            out.write(frame)
        except Exception as e:
            print(f"Error processing frame {frame_nmr}: {str(e)}")
            continue

    # Clean up
    out.release()
    cap.release()
    print(f"Processing complete. Output saved to {output_video_path}")



# Example usage
if __name__ == "__main__":
    overlay_license_plate_on_video(
        video_path="input.mp4",
        results_csv_path="results.csv",
        output_video_path="output.mp4"
    )



# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=5, line_length_x=100, line_length_y=100):
#     x1, y1 = top_left
#     x2, y2 = bottom_right
    
#     # Calculate box size to determine thickness
#     box_width = x2 - x1
#     box_height = y2 - y1
#     box_size = (box_width + box_height) / 2
    
#     # Scale thickness based on box size (smaller boxes = thinner borders)
#     scale_factor = min(1.0, box_size / 300)  # Adjust 300 based on your typical box sizes
#     thickness = max(1, int(thickness * scale_factor))
#     line_length_x = max(10, int(line_length_x * scale_factor))
#     line_length_y = max(10, int(line_length_y * scale_factor))

#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
#     cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

#     return img

# def overlay_license_plate_on_video(
#     video_path: str,
#     results_csv_path: str,
#     output_video_path: str
# ):
#     # Load results CSV
#     results = pd.read_csv(results_csv_path)

#     # Load video
#     cap = cv2.VideoCapture(video_path)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     # Prepare license plates (only for cars that have them)
#     license_plate = {}
#     cars_with_plates = results[results['license_number'].notna()]
    
#     for car_id in np.unique(cars_with_plates['car_id']):
#         # Get the frame with highest score license plate for this car
#         max_score = np.amax(cars_with_plates[cars_with_plates['car_id'] == car_id]['license_number_score'])
#         best_row = cars_with_plates[
#             (cars_with_plates['car_id'] == car_id) & 
#             (cars_with_plates['license_number_score'] == max_score)
#         ].iloc[0]
        
#         license_plate[car_id] = {
#             'license_crop': None,
#             'license_plate_number': best_row['license_number']
#         }

#         cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
#         ret, frame = cap.read()

#         if ret and best_row['license_plate_bbox'] != 'None':
#             try:
#                 bbox_str = best_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#                 x1, y1, x2, y2 = ast.literal_eval(bbox_str)
                
#                 # Validate bounding box coordinates
#                 if x2 <= x1 or y2 <= y1:
#                     print(f"Invalid license plate bbox for car {car_id}: [{x1}, {y1}, {x2}, {y2}]")
#                     continue
                
#                 # Ensure coordinates are within frame bounds
#                 h, w = frame.shape[:2]
#                 x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                
#                 if x2 <= x1 or y2 <= y1:
#                     print(f"Adjusted license plate bbox invalid for car {car_id}: [{x1}, {y1}, {x2}, {y2}]")
#                     continue

#                 license_crop = frame[y1:y2, x1:x2]
                
#                 # Calculate new width maintaining aspect ratio
#                 crop_height = y2 - y1
#                 if crop_height == 0:
#                     print(f"Zero height license plate for car {car_id}")
#                     continue
                
#                 new_width = int((x2 - x1) * 400 / crop_height)
#                 if new_width <= 0:
#                     print(f"Invalid calculated width for car {car_id}: {new_width}")
#                     continue
                
#                 license_crop = cv2.resize(license_crop, (new_width, 400))
#                 license_plate[car_id]['license_crop'] = license_crop
#             except Exception as e:
#                 print(f"Error processing license plate for car {car_id}: {str(e)}")
#                 continue

#     # Reset video
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     frame_nmr = -1
#     ret = True

#     while ret:
#         ret, frame = cap.read()
#         frame_nmr += 1

#         if ret:
#             df_frame = results[results['frame_nmr'] == frame_nmr]

#             for _, row in df_frame.iterrows():
#                 try:
#                     # Get car bounding box
#                     car_bbox_str = row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#                     car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str)
#                     car_id = row['car_id']
                    
#                     # Determine if this car has a license plate
#                     has_plate = car_id in license_plate and row['license_plate_bbox'] != 'None'
                    
#                     # Adjust box thickness based on distance (box size) and plate detection
#                     box_width = car_x2 - car_x1
#                     box_height = car_y2 - car_y1
#                     box_size = (box_width + box_height) / 2
                    
#                     # Base thickness - thicker if plate detected
#                     base_thickness = 12 if has_plate else 5
                    
#                     # Scale thickness based on box size
#                     scale_factor = min(1.0, box_size / 300)
#                     thickness = max(2, int(base_thickness * scale_factor))
                    
#                     # Draw car bounding box with adjusted thickness
#                     draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), 
#                                (0, 255, 0) if has_plate else (0, 165, 255),  # Green if plate, orange if no plate
#                                thickness)
                    
#                     # Only process license plate if this car has one
#                     if has_plate:
#                         try:
#                             # Draw license plate bounding box (thicker for plates)
#                             lp_bbox_str = row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#                             x1, y1, x2, y2 = ast.literal_eval(lp_bbox_str)
#                             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), max(3, thickness//2))

#                             # Overlay license plate crop
#                             license_crop = license_plate[car_id]['license_crop']
#                             if license_crop is not None:
#                                 H, W, _ = license_crop.shape

#                                 # Scale overlay size based on car distance
#                                 overlay_scale = min(1.0, box_size / 200)
#                                 display_h = int(H * overlay_scale)
#                                 display_w = int(W * overlay_scale)

#                                 if display_h > 0 and display_w > 0:
#                                     license_crop_scaled = cv2.resize(license_crop, (display_w, display_h))

#                                     # Place cropped license plate image
#                                     try:
#                                         y_start = int(car_y1) - display_h - int(50 * overlay_scale)
#                                         y_end = int(car_y1) - int(50 * overlay_scale)
#                                         x_center = int((car_x2 + car_x1) / 2)
#                                         x_start = x_center - display_w // 2
#                                         x_end = x_center + display_w // 2

#                                         # Handle potential 1-pixel mismatch
#                                         y_end = min(y_end, frame.shape[0])  # Ensure y_end doesn't exceed the frame height
#                                         x_end = min(x_end, frame.shape[1])  # Ensure x_end doesn't exceed the frame width
#                                         y_start = max(y_start, 0)  # Ensure y_start isn't negative
#                                         x_start = max(x_start, 0)  # Ensure x_start isn't negative

#                                         # If the calculated region exceeds the frame dimensions, we adjust to fit
#                                         frame[y_start:y_end, x_start:x_end] = license_crop_scaled[:(y_end - y_start), :(x_end - x_start)]

#                                     except Exception as e:
#                                         print(f"Error overlaying license plate image for car {car_id}: {str(e)}")
#                                         continue

#                         except Exception as e:
#                             print(f"Error drawing license plate for car {car_id}: {str(e)}")
#                             continue
#                 except Exception as e:
#                     print(f"Error processing car in frame {frame_nmr}: {str(e)}")
#                     continue

#             out.write(frame)

#     out.release()
#     cap.release()

