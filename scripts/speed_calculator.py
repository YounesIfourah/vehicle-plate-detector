import cv2
from collections import defaultdict

# Constants
Y1_PERCENT = 0.3
Y2_PERCENT = 0.7
LINE_OFFSET_PERCENT = 0.01
FPS = 25
DISTANCES = [40, 20, 10, 10]  # Distances between consecutive lines in meters
PRECISION = len(DISTANCES)  # Number of intermediate lines (one less than number of segments)

def calculate_segment_speed(frame_diff, segment_distance):
    time_seconds = frame_diff / FPS
    return round((segment_distance / time_seconds) * 3.6, 2)

def store_speed_to_results(results, car_id, start_frame, end_frame, speed, direction):
    for f in range(start_frame, end_frame + 1):
        if f not in results:
            continue
        if car_id not in results[f]:
            continue
        results[f][car_id]['speed'] = speed
        results[f][car_id]['direction'] = direction
        if 'segment_speeds' not in results[f][car_id]:
            results[f][car_id]['segment_speeds'] = []
        results[f][car_id]['segment_speeds'].append(speed)

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

    # Setup video
    cap = cv2.VideoCapture('./sample.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return results

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('back.mp4', fourcc, fps, (frame_width, frame_height))

    frame_nmr = -1
    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Draw lines
        for line_y in y_lines:
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, line_y - line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Process vehicle positions
        if frame_nmr in results:
            for car_id, car_data in results[frame_nmr].items():
                cy = car_data['car']['center'][1]
                centers[car_id] = cy

                # Record line crossings
                for i, line_y in enumerate(y_lines):
                    if crossings[car_id][i] is not None:
                        continue
                    in_zone = (line_y - line_offset) <= cy <= (line_y + line_offset)
                    if in_zone:
                        crossings[car_id][i] = frame_nmr

                # Display speed
                if 'speed' in car_data:
                    x1, y1, x2, y2 = car_data['car']['bbox']
                    info = f"ID:{car_id} {car_data['speed']:.1f}km/h {car_data.get('direction', '')}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, info, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)
        cv2.imshow('Vehicle Speed Tracking', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate speed between every valid pair of crossings for each vehicle
    for car_id, line_frames in crossings.items():
        for i in range(len(line_frames) - 1):
            for j in range(i + 1, len(line_frames)):
                f1 = line_frames[i]
                f2 = line_frames[j]
                if f1 is None or f2 is None:
                    continue
                
                # Calculate total distance between the lines
                if i < j:
                    segment_distances = DISTANCES[i:j]
                else:
                    segment_distances = DISTANCES[j:i]
                total_distance = sum(segment_distances)
                
                if f1 < f2:
                    frame_diff = f2 - f1
                    speed = calculate_segment_speed(frame_diff, total_distance)
                    direction = 'down'
                    store_speed_to_results(results, car_id, f1, f2, speed, direction)
                elif f1 > f2:
                    frame_diff = f1 - f2
                    speed = calculate_segment_speed(frame_diff, total_distance)
                    direction = 'up'
                    store_speed_to_results(results, car_id, f2, f1, speed, direction)

    return results



# # Constants
# Y1_PERCENT = 0.3
# Y2_PERCENT = 0.7
# LINE_OFFSET_PERCENT = 0.01
# FPS = 25
# DISTANCE_METERS = 80
# PRECISION = 3

# def calculate_segment_speed(frame_diff):
#     time_seconds = frame_diff / FPS
#     segment_distance = DISTANCE_METERS / PRECISION
#     return round((segment_distance / time_seconds) * 3.6, 2)

# def store_speed_to_results(results, car_id, start_frame, end_frame, speed, direction):
#     for f in range(start_frame, end_frame + 1):
#         if f not in results:
#             continue
#         if car_id not in results[f]:
#             continue
#         results[f][car_id]['speed'] = speed
#         results[f][car_id]['direction'] = direction
#         if 'segment_speeds' not in results[f][car_id]:
#             results[f][car_id]['segment_speeds'] = []
#         results[f][car_id]['segment_speeds'].append(speed)

# def calculate_speed(results, frame_height):
#     # Calculate line positions
#     y_lines = []
#     for i in range(PRECISION + 1):
#         percent = Y1_PERCENT + (Y2_PERCENT - Y1_PERCENT) * i / PRECISION
#         y_lines.append(int(frame_height * percent))
#     line_offset = int(frame_height * LINE_OFFSET_PERCENT)

#     # Data structures
#     crossings = defaultdict(lambda: [None] * (PRECISION + 1))
#     centers = {}

#     # Setup video
#     cap = cv2.VideoCapture('./sample.mp4')
#     if not cap.isOpened():
#         print("Error: Could not open video file")
#         return results

#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('back.mp4', fourcc, fps, (frame_width, frame_height))

#     frame_nmr = -1
#     while True:
#         frame_nmr += 1
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Draw lines
#         for line_y in y_lines:
#             cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, line_y - line_offset), (frame.shape[1], line_y + line_offset), (0, 200, 200), -1)
#             cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

#         # Process vehicle positions
#         if frame_nmr in results:
#             for car_id, car_data in results[frame_nmr].items():
#                 cy = car_data['car']['center'][1]
#                 centers[car_id] = cy

#                 # Record line crossings
#                 for i, line_y in enumerate(y_lines):
#                     if crossings[car_id][i] is not None:
#                         continue
#                     in_zone = (line_y - line_offset) <= cy <= (line_y + line_offset)
#                     if in_zone:
#                         crossings[car_id][i] = frame_nmr

#                 # Display speed
#                 if 'speed' in car_data:
#                     x1, y1, x2, y2 = car_data['car']['bbox']
#                     info = f"ID:{car_id} {car_data['speed']:.1f}km/h {car_data.get('direction', '')}"
#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#                     cv2.putText(frame, info, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

#         out.write(frame)
#         cv2.imshow('Vehicle Speed Tracking', frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # Calculate speed between every valid pair of crossings for each vehicle
#     for car_id, line_frames in crossings.items():
#         for i in range(len(line_frames)):
#             for j in range(len(line_frames)):
#                 if i == j:
#                     continue
#                 f1 = line_frames[i]
#                 f2 = line_frames[j]
#                 if f1 is None or f2 is None:
#                     continue
#                 if f1 < f2:
#                     frame_diff = f2 - f1
#                     speed = calculate_segment_speed(frame_diff)
#                     direction = 'down'
#                     store_speed_to_results(results, car_id, f1, f2, speed, direction)
#                 elif f1 > f2:
#                     frame_diff = f1 - f2
#                     speed = calculate_segment_speed(frame_diff)
#                     direction = 'up'
#                     store_speed_to_results(results, car_id, f2, f1, speed, direction)

#     return results

