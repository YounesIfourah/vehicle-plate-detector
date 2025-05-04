import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    car_centers = np.array([list(map(float, row['car_center'][1:-1].split())) for row in data])
    
    # Handle speed (preserve None values)
    speeds = []
    for row in data:
        if 'speed' not in row or row['speed'] in ['None', '', None]:
            speeds.append(None)  # Keep as None instead of 0.0
        else:
            try:
                speeds.append(float(row['speed']))
            except (ValueError, TypeError):
                speeds.append(None)  # Keep as None instead of 0.0
    speeds = np.array(speeds)
    
    # Handle license plate bboxes (can be 'None')
    license_plate_bboxes = []
    for row in data:
        if row['license_plate_bbox'] == 'None':
            license_plate_bboxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            license_plate_bboxes.append(list(map(float, row['license_plate_bbox'][1:-1].split())))
    license_plate_bboxes = np.array(license_plate_bboxes)

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        car_centers_interpolated = []
        license_plate_bboxes_interpolated = []
        speeds_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            car_center = car_centers[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]
            speed = speeds[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_car_center = car_centers_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
                prev_speed = speeds_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)

                    # Interpolate car bbox
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    
                    # Interpolate car center
                    interp_func = interp1d(x, np.vstack((prev_car_center, car_center)), axis=0, kind='linear')
                    interpolated_car_centers = interp_func(x_new)

                    # Only interpolate speed if both values are not None
                    if prev_speed is not None and speed is not None:
                        interp_func = interp1d(x, [prev_speed, speed], kind='linear')
                        interpolated_speeds = interp_func(x_new)
                    else:
                        interpolated_speeds = [None] * len(x_new)

                    # Only interpolate license plate if both frames have plates
                    if not np.all(prev_license_plate_bbox == 0) and not np.all(license_plate_bbox == 0):
                        interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                        interpolated_license_plate_bboxes = interp_func(x_new)
                    else:
                        interpolated_license_plate_bboxes = np.zeros((len(x_new), 4))

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    car_centers_interpolated.extend(interpolated_car_centers[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
                    speeds_interpolated.extend(interpolated_speeds[1:])

            car_bboxes_interpolated.append(car_bbox)
            car_centers_interpolated.append(car_center)
            license_plate_bboxes_interpolated.append(license_plate_bbox)
            speeds_interpolated.append(speed)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            # Convert None to 'None' when creating the row
            speed_value = 'None' if speeds_interpolated[i] is None else str(speeds_interpolated[i])
            
            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': '[' + ' '.join(map(str, car_bboxes_interpolated[i])) + ']',
                'car_center': '[' + ' '.join(map(str, car_centers_interpolated[i])) + ']',
                'license_plate_bbox': '[' + ' '.join(map(str, license_plate_bboxes_interpolated[i])) + ']',
                'speed': speed_value,
            }

            if str(frame_number) not in frame_numbers_:
                # For interpolated frames, set all license plate fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # For original frames, copy all values from the original data
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data

def interpolate_vehicle_tracking_csv(input_csv_path, output_csv_path):
    """
    Load vehicle tracking CSV, interpolate missing frames, and save the new data into a CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the interpolated CSV file.
    """
    with open(input_csv_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    interpolated_data = interpolate_bounding_boxes(data)

    header = [
        'frame_nmr', 'car_id', 'score', 'car_bbox', 'car_center', 'license_plate_bbox',
        'license_plate_bbox_score', 'license_number', 'license_number_score', 'speed'
    ]
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)


        
# import csv
# import numpy as np
# from scipy.interpolate import interp1d

# def interpolate_bounding_boxes(data):
#     frame_numbers = np.array([int(row['frame_nmr']) for row in data])
#     car_ids = np.array([int(float(row['car_id'])) for row in data])
#     car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
#     car_centers = np.array([list(map(float, row['car_center'][1:-1].split())) for row in data])
    
#     # Handle speed (can be missing, 'None', or empty)
#     speeds = []
#     for row in data:
#         if 'speed' not in row or row['speed'] in ['None', '', None]:
#             speeds.append(0.0)
#         else:
#             try:
#                 speeds.append(float(row['speed']))
#             except (ValueError, TypeError):
#                 speeds.append(0.0)
#     speeds = np.array(speeds)
    
#     # Handle license plate bboxes (can be 'None')
#     license_plate_bboxes = []
#     for row in data:
#         if row['license_plate_bbox'] == 'None':
#             license_plate_bboxes.append([0.0, 0.0, 0.0, 0.0])
#         else:
#             license_plate_bboxes.append(list(map(float, row['license_plate_bbox'][1:-1].split())))
#     license_plate_bboxes = np.array(license_plate_bboxes)

#     interpolated_data = []
#     unique_car_ids = np.unique(car_ids)
#     for car_id in unique_car_ids:
#         frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

#         car_mask = car_ids == car_id
#         car_frame_numbers = frame_numbers[car_mask]
#         car_bboxes_interpolated = []
#         car_centers_interpolated = []
#         license_plate_bboxes_interpolated = []
#         speeds_interpolated = []

#         first_frame_number = car_frame_numbers[0]

#         for i in range(len(car_bboxes[car_mask])):
#             frame_number = car_frame_numbers[i]
#             car_bbox = car_bboxes[car_mask][i]
#             car_center = car_centers[car_mask][i]
#             license_plate_bbox = license_plate_bboxes[car_mask][i]
#             speed = speeds[car_mask][i]

#             if i > 0:
#                 prev_frame_number = car_frame_numbers[i-1]
#                 prev_car_bbox = car_bboxes_interpolated[-1]
#                 prev_car_center = car_centers_interpolated[-1]
#                 prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
#                 prev_speed = speeds_interpolated[-1]

#                 if frame_number - prev_frame_number > 1:
#                     frames_gap = frame_number - prev_frame_number
#                     x = np.array([prev_frame_number, frame_number])
#                     x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)

#                     # Interpolate car bbox
#                     interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
#                     interpolated_car_bboxes = interp_func(x_new)
                    
#                     # Interpolate car center
#                     interp_func = interp1d(x, np.vstack((prev_car_center, car_center)), axis=0, kind='linear')
#                     interpolated_car_centers = interp_func(x_new)

#                     # Interpolate speed (simple linear interpolation)
#                     interp_func = interp1d(x, [prev_speed, speed], kind='linear')
#                     interpolated_speeds = interp_func(x_new)

#                     # Only interpolate license plate if both frames have plates
#                     if not np.all(prev_license_plate_bbox == 0) and not np.all(license_plate_bbox == 0):
#                         interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
#                         interpolated_license_plate_bboxes = interp_func(x_new)
#                     else:
#                         interpolated_license_plate_bboxes = np.zeros((len(x_new), 4))

#                     car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
#                     car_centers_interpolated.extend(interpolated_car_centers[1:])
#                     license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
#                     speeds_interpolated.extend(interpolated_speeds[1:])

#             car_bboxes_interpolated.append(car_bbox)
#             car_centers_interpolated.append(car_center)
#             license_plate_bboxes_interpolated.append(license_plate_bbox)
#             speeds_interpolated.append(speed)

#         for i in range(len(car_bboxes_interpolated)):
#             frame_number = first_frame_number + i
#             row = {
#                 'frame_nmr': str(frame_number),
#                 'car_id': str(car_id),
#                 'car_bbox': '[' + ' '.join(map(str, car_bboxes_interpolated[i])) + ']',
#                 'car_center': '[' + ' '.join(map(str, car_centers_interpolated[i])) + ']',
#                 'license_plate_bbox': '[' + ' '.join(map(str, license_plate_bboxes_interpolated[i])) + ']',
#                 'speed': str(speeds_interpolated[i]),
#             }

#             if str(frame_number) not in frame_numbers_:
#                 # For interpolated frames, set all license plate fields to '0'
#                 row['license_plate_bbox_score'] = '0'
#                 row['license_number'] = '0'
#                 row['license_number_score'] = '0'
#             else:
#                 # For original frames, copy all values from the original data
#                 original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
#                 row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
#                 row['license_number'] = original_row.get('license_number', '0')
#                 row['license_number_score'] = original_row.get('license_number_score', '0')

#             interpolated_data.append(row)

#     return interpolated_data

# def interpolate_vehicle_tracking_csv(input_csv_path, output_csv_path):
#     """
#     Load vehicle tracking CSV, interpolate missing frames, and save the new data into a CSV file.

#     Args:
#         input_csv_path (str): Path to the input CSV file.
#         output_csv_path (str): Path to save the interpolated CSV file.
#     """
#     with open(input_csv_path, 'r') as file:
#         reader = csv.DictReader(file)
#         data = list(reader)

#     interpolated_data = interpolate_bounding_boxes(data)

#     header = [
#         'frame_nmr', 'car_id', 'score', 'car_bbox', 'car_center', 'license_plate_bbox',
#         'license_plate_bbox_score', 'license_number', 'license_number_score', 'speed'
#     ]
#     with open(output_csv_path, 'w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=header)
#         writer.writeheader()
#         writer.writerows(interpolated_data)


