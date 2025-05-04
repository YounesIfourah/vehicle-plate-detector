import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        # Updated header to include speed
        f.write('frame_nmr,car_id,car_bbox,car_center,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,speed\n')

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                car_data = results[frame_nmr][car_id]
                
                # Get car bbox and center (should always exist)
                car_bbox = car_data['car']['bbox']
                car_center = car_data['car']['center']
                
                # Get license plate data (might be None)
                lp_data = car_data['license_plate']
                lp_bbox = lp_data['bbox'] if lp_data['bbox'] is not None else [None, None, None, None]
                lp_bbox_score = lp_data['bbox_score'] if lp_data['bbox_score'] is not None else "None"
                lp_text = lp_data['text'] if lp_data['text'] is not None else "None"
                lp_text_score = lp_data['text_score'] if lp_data['text_score'] is not None else "None"
                
                # Get speed (might not exist)
                speed = car_data.get('speed', "None")

                # Format the bbox strings
                car_bbox_str = f"[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]"
                car_center_str = f"[{car_center[0]} {car_center[1]}]"
                lp_bbox_str = f"[{lp_bbox[0]} {lp_bbox[1]} {lp_bbox[2]} {lp_bbox[3]}]" if lp_bbox[0] is not None else "None"

                # Write the line
                f.write(f"{frame_nmr},{car_id},{car_bbox_str},{car_center_str},{lp_bbox_str},{lp_bbox_score},{lp_text},{lp_text_score},{speed}\n")

                


# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file.

#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         # Write header
#         f.write('frame_nmr,car_id,car_bbox,car_center,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')

#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 car_data = results[frame_nmr][car_id]
                
#                 # Get car bbox and center (should always exist)
#                 car_bbox = car_data['car']['bbox']
#                 car_center = car_data['car']['center']
                
#                 # Get license plate data (might be None)
#                 lp_data = car_data['license_plate']
#                 lp_bbox = lp_data['bbox'] if lp_data['bbox'] is not None else [None, None, None, None]
#                 lp_bbox_score = lp_data['bbox_score']
#                 lp_text = lp_data['text']
#                 lp_text_score = lp_data['text_score']

#                 # Format the bbox strings
#                 car_bbox_str = f"[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]"
#                 car_center_str = f"[{car_center[0]} {car_center[1]}]"
#                 lp_bbox_str = f"[{lp_bbox[0]} {lp_bbox[1]} {lp_bbox[2]} {lp_bbox[3]}]" if lp_bbox[0] is not None else "None"

#                 # Write the line
#                 f.write(f"{frame_nmr},{car_id},{car_bbox_str},{car_center_str},{lp_bbox_str},{lp_bbox_score},{lp_text},{lp_text_score}\n")



                
    # No need for explicit f.close() - the 'with' statement handles it



# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file.

#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
#                                                 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#                                                 'license_number_score'))

#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 print(results[frame_nmr][car_id])
#                 if 'car' in results[frame_nmr][car_id].keys() and \
#                    'license_plate' in results[frame_nmr][car_id].keys() and \
#                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
#                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                             car_id,
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['car']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][3]),
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][3]),
#                                                             results[frame_nmr][car_id]['license_plate']['bbox_score'],
#                                                             results[frame_nmr][car_id]['license_plate']['text'],
#                                                             results[frame_nmr][car_id]['license_plate']['text_score'])
#                             )
#         f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car_id_given_a_plate(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def get_car_id_given_a_plate(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1