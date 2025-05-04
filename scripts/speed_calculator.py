import cv2
import numpy as np
from collections import defaultdict

# Constants
Y1_PERCENT = 0.3
Y2_PERCENT = 0.95
LINE_OFFSET_PERCENT = 0.01
FPS = 25
DISTANCES = [40, 20, 10, 7, 4]  # Distances between consecutive lines in meters
PRECISION = len(DISTANCES)  # Number of intermediate lines (one less than number of segments)

def calculate_segment_speed(frame_diff, segment_distance):
    time_seconds = frame_diff / FPS
    return round((segment_distance / time_seconds) * 3.6, 2)

def store_speed_to_results(results, car_id, start_frame, end_frame, start_speed, end_speed, direction):
    # Store linear speed change between start and end frames
    for f in range(start_frame, end_frame + 1):
        if f not in results:
            continue
        if car_id not in results[f]:
            continue
        
        # Linear interpolation of speed
        progress = (f - start_frame) / (end_frame - start_frame)
        current_speed = start_speed + (end_speed - start_speed) * progress
        
        results[f][car_id]['speed'] = round(current_speed, 2)
        results[f][car_id]['direction'] = direction
        
        # Store original calculated speeds for reference
        if 'calculated_speeds' not in results[f][car_id]:
            results[f][car_id]['calculated_speeds'] = []
        if f == start_frame or f == end_frame:
            results[f][car_id]['calculated_speeds'].append(round(current_speed, 2))

def calculate_speed(results, frame_height):
    # Calculate line positions
    y_lines = []
    for i in range(PRECISION + 1):
        percent = Y1_PERCENT + (Y2_PERCENT - Y1_PERCENT) * i / PRECISION
        y_lines.append(int(frame_height * percent))
    line_offset = int(frame_height * LINE_OFFSET_PERCENT)

    # Data structures
    crossings = defaultdict(lambda: [None] * (PRECISION + 1))
    centers = {}
    speed_points = defaultdict(list)  # To store all speed measurement points per car

    # Setup video
    cap = cv2.VideoCapture('./sample.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return results

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create resizable window
    cv2.namedWindow('Vehicle Speed Tracking', cv2.WINDOW_NORMAL)
    
    # Calculate display scaling (max 80% of screen height)
    screen_height = 1080
    max_display_height = int(screen_height * 0.8)
    scale_factor = min(1.0, max_display_height / frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('back.mp4', fourcc, fps, (frame_width, frame_height))

    frame_nmr = -1
    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Draw measurement zone lines
        for line_y in y_lines:
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, line_y - line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Process vehicle positions
        if frame_nmr in results:
            for car_id, car_data in results[frame_nmr].items():
                bbox = car_data['car']['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cy = car_data['car']['center'][1]
                centers[car_id] = cy

                # Record line crossings
                for i, line_y in enumerate(y_lines):
                    if crossings[car_id][i] is not None:
                        continue
                    in_zone = (line_y - line_offset) <= cy <= (line_y + line_offset)
                    if in_zone:
                        crossings[car_id][i] = frame_nmr
                        # Store crossing point with line index
                        speed_points[car_id].append((frame_nmr, i))

                # Draw bounding box (thicker when speed is available)
                box_thickness = 3 if 'speed' in car_data else 1
                box_color = (0, 0, 255) if 'speed' in car_data else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
                
                # Draw vehicle ID above the bounding box
                id_text = f"ID:{car_id}"
                cv2.putText(frame, id_text, (x1, y1 - 5 if y1 > 30 else y2 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Display speed if available
                if 'speed' in car_data:
                    speed_text = f"{car_data['speed']:.1f} km/h"
                    (text_width, text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # Create background rectangle
                    cv2.rectangle(frame, 
                                (x1, y1 - text_height - 10), 
                                (x1 + text_width + 10, y1 - 5), 
                                (40, 40, 40), -1)
                    
                    # Draw speed text
                    cv2.putText(frame, speed_text, (x1 + 5, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    
                    # Draw direction arrow
                    direction = car_data.get('direction', '')
                    arrow_y = y1 - text_height - 25 if y1 > 50 else y2 + text_height + 5
                    if direction == 'up':
                        cv2.arrowedLine(frame, (x1, arrow_y), (x1 + 20, arrow_y - 20), (0, 255, 0), 2)
                    elif direction == 'down':
                        cv2.arrowedLine(frame, (x1, arrow_y), (x1 + 20, arrow_y + 20), (0, 255, 0), 2)

        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_nmr}", (frame_width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Resize for display if needed
        if scale_factor != 1.0:
            display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        else:
            display_frame = frame
            
        out.write(frame)
        cv2.imshow('Vehicle Speed Tracking', display_frame)
        
        # Keyboard controls
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause on space
            while True:
                key = cv2.waitKey(0)
                if key == ord(' '):  # Unpause on space
                    break
                elif key == ord('q'):  # Quit on q
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate speeds between measurement points with linear interpolation
    for car_id, points in speed_points.items():
        # Sort points by frame number
        points.sort(key=lambda x: x[0])
        
        # Calculate speeds between consecutive measurement points
        for i in range(len(points) - 1):
            frame1, line_idx1 = points[i]
            frame2, line_idx2 = points[i + 1]
            
            # Calculate distance between these measurement points
            if line_idx1 < line_idx2:
                segment_distances = DISTANCES[line_idx1:line_idx2]
            else:
                segment_distances = DISTANCES[line_idx2:line_idx1]
            total_distance = sum(segment_distances)
            
            frame_diff = abs(frame2 - frame1)
            calculated_speed = calculate_segment_speed(frame_diff, total_distance)
            
            # Determine direction
            direction = 'down' if line_idx1 < line_idx2 else 'up'
            
            # If this is the first segment, we need to set initial speed
            if i == 0:
                # For the first segment, assume constant speed until first measurement
                for f in range(0, frame1 + 1):
                    if f in results and car_id in results[f]:
                        results[f][car_id]['speed'] = calculated_speed
                        results[f][car_id]['direction'] = direction
            
            # Get previous speed for interpolation
            prev_speed = results[frame1][car_id]['speed'] if frame1 in results and car_id in results[frame1] and 'speed' in results[frame1][car_id] else calculated_speed
            
            # Store with linear interpolation
            store_speed_to_results(results, car_id, frame1, frame2, prev_speed, calculated_speed, direction)
            
            # For the last segment, maintain speed after last measurement
            if i == len(points) - 2:
                last_frame = max(results.keys())
                for f in range(frame2, last_frame + 1):
                    if f in results and car_id in results[f]:
                        results[f][car_id]['speed'] = calculated_speed
                        results[f][car_id]['direction'] = direction

    return results

