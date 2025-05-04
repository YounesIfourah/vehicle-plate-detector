from scripts import (
    vehicle_and_plate_tracking,
    interpolate_vehicle_tracking_csv,
    process_video
)


def main():
    # Paths
    input_video_path = './sample.mp4'
    raw_output_csv = './test.csv'
    interpolated_output_csv = './test_interpolated.csv'
    output_video_path = './out.mp4'
    
    # Step 1: Track vehicles and plates, save to CSV
    vehicle_and_plate_tracking(
        video_path=input_video_path,
        output_csv_path=raw_output_csv
    )

    # Step 2: Interpolate missing vehicle tracking data
    interpolate_vehicle_tracking_csv(
        raw_output_csv,
        interpolated_output_csv
    )

    # Step 3: Overlay license plates on the video and save the final video
    process_video(
        video_path=input_video_path,
        results_csv_path=interpolated_output_csv,
        output_video_path=output_video_path
    )

if __name__ == "__main__":
    main()
