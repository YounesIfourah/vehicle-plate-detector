import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def overlay_license_plate_on_video(
    video_path: str,
    results_csv_path: str,
    output_video_path: str
):
    # Load results CSV
    results = pd.read_csv(results_csv_path)

    # Load video
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Prepare license plates
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        best_row = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)].iloc[0]
        
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_number': best_row['license_number']
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
        ret, frame = cap.read()

        if not ret:
            continue

        bbox_str = best_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        x1, y1, x2, y2 = ast.literal_eval(bbox_str)

        license_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        license_plate[car_id]['license_crop'] = license_crop

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_nmr = -1
    ret = True

    while ret:
        ret, frame = cap.read()
        frame_nmr += 1

        if ret:
            df_frame = results[results['frame_nmr'] == frame_nmr]

            for _, row in df_frame.iterrows():
                # Draw car bounding box
                car_bbox_str = row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str)
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25)

                # Draw license plate bounding box
                lp_bbox_str = row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                x1, y1, x2, y2 = ast.literal_eval(lp_bbox_str)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # Overlay license plate crop
                car_id = row['car_id']
                license_crop = license_plate[car_id]['license_crop']

                if license_crop is None:
                    continue

                H, W, _ = license_crop.shape

                try:
                    # Place cropped license plate image
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = license_crop

                    # White background for license number text
                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = (255, 255, 255)

                    # Write license number
                    text = license_plate[car_id]['license_plate_number']
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                    
                    cv2.putText(frame,
                                text,
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)
                except Exception as e:
                    print(f"Error overlaying license plate for car_id {car_id}: {e}")
                    continue

            out.write(frame)

    out.release()
    cap.release()



