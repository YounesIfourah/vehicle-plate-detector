# Global variables for speed calculation
Y1_PERCENT = 0.3  # First reference line position (30% from top)
Y2_PERCENT = 0.7    # Second reference line position (70% from top)
LINE_OFFSET_PERCENT = 0.02  # Percentage tolerance for line crossing detection
FPS = 25            # Frames per second of the video
DISTANCE_METERS = 80 # Actual distance between Y1 and Y2 lines in meters
PRECISION = 2        # Number of segments between Y1 and Y2 (1 = original version)

def calculate_speed(results, frame_height):
    """
    Calculate vehicle speeds with multiple reference lines for more precise tracking.
    Uses the global PRECISION variable to determine how many segments to create.
    
    Args:
        results: Detection results
        frame_height: Height of the video frame
    
    Returns:
        Modified results with speed information
    """
    global PRECISION  # Use the global precision value
    
    # Calculate line positions
    y_start = int(frame_height * Y1_PERCENT)
    y_end = int(frame_height * Y2_PERCENT)
    
    # Create equally spaced lines based on precision
    line_positions = [y_start + int((y_end - y_start) * i/PRECISION) for i in range(PRECISION + 1)]
    line_offset = int(frame_height * LINE_OFFSET_PERCENT)
    
    # For each car, we'll track which segments it has crossed
    tracking = {}  # Format: {car_id: {'segment_states': [state_for_each_segment], 'crossing_info': []}}
    previous_centers = {}

    # Visualization setup
    import cv2
    cap = cv2.VideoCapture('./sample.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return results
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('back.mp4', fourcc, fps, (frame_width, frame_height))
    
    cv2.namedWindow('Vehicle Speed Tracking', cv2.WINDOW_NORMAL)
    
    frame_nmr = -1
    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw reference lines and their offset zones
        for line_y in line_positions:
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
            cv2.line(frame, (0, line_y - line_offset), (frame.shape[1], line_y - line_offset), (0, 200, 200), 1)
            cv2.line(frame, (0, line_y + line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, line_y - line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        if frame_nmr in results:
            for car_id, car_data in results[frame_nmr].items():
                x1, y1, x2, y2 = car_data['car']['bbox']
                cy = car_data['car']['center'][1]
                
                # Save previous center
                prev_cy = previous_centers.get(car_id, cy)
                previous_centers[car_id] = cy

                # Initialize tracking for this car if not exists
                if car_id not in tracking:
                    tracking[car_id] = {
                        'segment_states': [None] * PRECISION,  # Tracks state for each segment
                        'crossing_info': [],  # Stores tuples of (start_frame, start_line_idx)
                        'speeds': []  # Stores calculated speeds for each segment
                    }
                
                # Check each segment for crossing
                for segment_idx in range(PRECISION):
                    upper_line = line_positions[segment_idx]
                    lower_line = line_positions[segment_idx + 1]
                    
                    # Check if we're entering the upper line's zone
                    if tracking[car_id]['segment_states'][segment_idx] is None:
                        was_outside = prev_cy < (upper_line - line_offset) or prev_cy > (upper_line + line_offset)
                        is_inside = (upper_line - line_offset) <= cy <= (upper_line + line_offset)
                        if was_outside and is_inside:
                            # Starting to cross this segment
                            tracking[car_id]['segment_states'][segment_idx] = 'crossing'
                            tracking[car_id]['crossing_info'].append((frame_nmr, segment_idx))
                    
                    # Check if we're entering the lower line's zone while crossing
                    elif tracking[car_id]['segment_states'][segment_idx] == 'crossing':
                        was_outside = prev_cy < (lower_line - line_offset) or prev_cy > (lower_line + line_offset)
                        is_inside = (lower_line - line_offset) <= cy <= (lower_line + line_offset)
                        if was_outside and is_inside:
                            # Finished crossing this segment
                            start_frame, start_segment_idx = tracking[car_id]['crossing_info'].pop(0)
                            frame_diff = frame_nmr - start_frame
                            time_seconds = frame_diff / FPS
                            segment_distance = DISTANCE_METERS / PRECISION  # Each segment is 1/PRECISION of total distance
                            speed_kph = round((segment_distance / time_seconds) * 3.6, 2)
                            
                            # Store the speed for this segment
                            tracking[car_id]['speeds'].append(speed_kph)
                            tracking[car_id]['segment_states'][segment_idx] = 'crossed'
                            
                            # Update the results with the latest speed
                            for fn in range(start_frame, frame_nmr + 1):
                                if fn in results and car_id in results[fn]:
                                    if 'speeds' not in results[fn][car_id]:
                                        results[fn][car_id]['speeds'] = []
                                    results[fn][car_id]['speeds'].append(speed_kph)
                                    results[fn][car_id]['latest_speed'] = speed_kph
                
                # Visualization
                color = (0, 255, 0)  # Default green
                # If currently crossing any segment, show red
                if 'crossing' in tracking[car_id]['segment_states']:
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                info = f"ID:{car_id}"
                if 'speeds' in car_data:
                    avg_speed = sum(car_data['speeds'])/len(car_data['speeds'])
                    info += f" {avg_speed:.1f}km/h"
                elif 'latest_speed' in car_data:
                    info += f" {car_data['latest_speed']}km/h"
                cv2.putText(frame, info, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out.write(frame)
        cv2.imshow('Vehicle Speed Tracking', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return results

# # Global variables for speed calculation
# Y1_PERCENT = 0.3  # First reference line position (30% from top)
# Y2_PERCENT = 0.7    # Second reference line position (70% from top)
# LINE_OFFSET_PERCENT = 0.02  # Percentage tolerance for line crossing detection
# FPS = 25            # Frames per second of the video
# DISTANCE_METERS = 80 # Actual distance between Y1 and Y2 lines in meters



# def calculate_speed(results, frame_height):
#     """
#     Calculate vehicle speeds by checking both possible line crossing orders separately.
#     Save the visualization as back.mp4.
#     """
#     y1_line = int(frame_height * Y1_PERCENT)
#     y2_line = int(frame_height * Y2_PERCENT)
#     line_offset = int(frame_height * LINE_OFFSET_PERCENT)
    
#     # Tracking dictionaries
#     y1_first = {}
#     y2_first = {}
#     previous_centers = {}

#     # Visualization setup
#     import cv2
#     cap = cv2.VideoCapture('./sample.mp4')
#     if not cap.isOpened():
#         print("Error: Could not open video file")
#         return results
    
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('back.mp4', fourcc, fps, (frame_width, frame_height))
    
#     cv2.namedWindow('Vehicle Speed Tracking', cv2.WINDOW_NORMAL)
    
#     frame_nmr = -1
#     while True:
#         frame_nmr += 1
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Draw reference lines
#         for line_y in [y1_line, y2_line]:
#             cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
#             cv2.line(frame, (0, line_y - line_offset), (frame.shape[1], line_y - line_offset), (0, 200, 200), 1)
#             cv2.line(frame, (0, line_y + line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), 1)
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, line_y - line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), -1)
#             cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
#         if frame_nmr in results:
#             for car_id, car_data in results[frame_nmr].items():
#                 x1, y1, x2, y2 = car_data['car']['bbox']
#                 cy = car_data['car']['center'][1]
                
#                 # Save previous center
#                 prev_cy = previous_centers.get(car_id, cy)
#                 previous_centers[car_id] = cy

#                 # Initialize tracking
#                 if car_id not in y1_first:
#                     y1_first[car_id] = {'y1_frame': None, 'y2_frame': None}
#                 if car_id not in y2_first:
#                     y2_first[car_id] = {'y2_frame': None, 'y1_frame': None}

#                 # Y1 → Y2 logic (trigger when entering Y1 offset)
#                 if y1_first[car_id]['y1_frame'] is None:
#                     was_outside = prev_cy < (y1_line - line_offset) or prev_cy > (y1_line + line_offset)
#                     is_inside = (y1_line - line_offset) <= cy <= (y1_line + line_offset)
#                     if was_outside and is_inside:
#                         y1_first[car_id]['y1_frame'] = frame_nmr

#                 if (y1_first[car_id]['y1_frame'] is not None and 
#                     y1_first[car_id]['y2_frame'] is None):
#                     was_outside = prev_cy < (y2_line - line_offset) or prev_cy > (y2_line + line_offset)
#                     is_inside = (y2_line - line_offset) <= cy <= (y2_line + line_offset)
#                     if was_outside and is_inside:
#                         y1_first[car_id]['y2_frame'] = frame_nmr
#                         frame_diff = y1_first[car_id]['y2_frame'] - y1_first[car_id]['y1_frame']
#                         time_seconds = frame_diff / FPS
#                         speed_kph = round((DISTANCE_METERS / time_seconds) * 3.6, 2)
#                         for fn in range(y1_first[car_id]['y1_frame'], y1_first[car_id]['y2_frame'] + 1):
#                             if fn in results and car_id in results[fn]:
#                                 results[fn][car_id]['speed'] = speed_kph
#                         y1_first[car_id] = {'y1_frame': None, 'y2_frame': None}

#                 # Y2 → Y1 logic (trigger when entering Y2 offset)
#                 if y2_first[car_id]['y2_frame'] is None:
#                     was_outside = prev_cy < (y2_line - line_offset) or prev_cy > (y2_line + line_offset)
#                     is_inside = (y2_line - line_offset) <= cy <= (y2_line + line_offset)
#                     if was_outside and is_inside:
#                         y2_first[car_id]['y2_frame'] = frame_nmr

#                 if (y2_first[car_id]['y2_frame'] is not None and 
#                     y2_first[car_id]['y1_frame'] is None):
#                     was_outside = prev_cy < (y1_line - line_offset) or prev_cy > (y1_line + line_offset)
#                     is_inside = (y1_line - line_offset) <= cy <= (y1_line + line_offset)
#                     if was_outside and is_inside:
#                         y2_first[car_id]['y1_frame'] = frame_nmr
#                         frame_diff = y2_first[car_id]['y1_frame'] - y2_first[car_id]['y2_frame']
#                         time_seconds = frame_diff / FPS
#                         speed_kph = round((DISTANCE_METERS / time_seconds) * 3.6, 2)
#                         for fn in range(y2_first[car_id]['y2_frame'], y2_first[car_id]['y1_frame'] + 1):
#                             if fn in results and car_id in results[fn]:
#                                 results[fn][car_id]['speed'] = speed_kph
#                         y2_first[car_id] = {'y2_frame': None, 'y1_frame': None}

#                 # Optional: reset tracking if far outside zone (not strictly required anymore)
#                 if (y2_first[car_id]['y2_frame'] is not None and 
#                     (cy < (y1_line - line_offset) or cy > (y2_line + line_offset))):
#                     y2_first[car_id] = {'y2_frame': None, 'y1_frame': None}

#                 # Visualization
#                 color = (0, 255, 0)
#                 if y2_first[car_id]['y2_frame'] is not None and y2_first[car_id]['y1_frame'] is None:
#                     color = (0, 0, 255)
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                 info = f"ID:{car_id}"
#                 if 'speed' in car_data:
#                     info += f" {car_data['speed']}km/h"
#                 cv2.putText(frame, info, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#         out.write(frame)
#         cv2.imshow('Vehicle Speed Tracking', frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     return results
