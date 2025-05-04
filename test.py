import cv2

# Configuration
VIDEO_PATH = 'sample.mp4'  # Update with your video path
TARGET_WIDTH = 1280            # 720p width
TARGET_HEIGHT = 720            # 720p height
LINE_COLOR = (0, 0, 255)       # Red color for lines
LINE_THICKNESS = 2             # Line thickness

# Initialize reference lines (will be adjusted based on resized video)
Y1_LINE = 300
Y2_LINE = 500

def resize_frame(frame, target_width, target_height):
    """Resize frame to target width while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    # Calculate new dimensions
    new_width = target_width
    new_height = int(new_width / aspect_ratio)
    
    # If the calculated height exceeds target, scale down
    if new_height > target_height:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    
    # Resize and pad if necessary to maintain 720p
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create black background
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return result, x_offset, y_offset, new_height

def visualize_reference_lines():
    global Y1_LINE, Y2_LINE
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get first frame to set initial line positions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        return
    
    # Get resized dimensions
    resized_frame, _, _, resized_height = resize_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
    
    # Set initial line positions based on resized video
    Y1_LINE = int(resized_height * 0.3)  # 40% from top
    Y2_LINE = int(resized_height * 0.7)  # 60% from top
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to first frame
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize the frame
            resized_frame, x_offset, y_offset, new_height = resize_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
            
            # Draw reference lines (adjusted for actual content area)
            actual_y1 = y_offset + int((Y1_LINE / TARGET_HEIGHT) * new_height)
            actual_y2 = y_offset + int((Y2_LINE / TARGET_HEIGHT) * new_height)
            
            cv2.line(resized_frame, (0, actual_y1), (TARGET_WIDTH, actual_y1), 
                    LINE_COLOR, LINE_THICKNESS)
            cv2.line(resized_frame, (0, actual_y2), (TARGET_WIDTH, actual_y2), 
                    LINE_COLOR, LINE_THICKNESS)
            
            # Add text labels
            cv2.putText(resized_frame, f"Y1: {Y1_LINE}", (10, actual_y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, LINE_COLOR, 2)
            cv2.putText(resized_frame, f"Y2: {Y2_LINE}", (10, actual_y2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, LINE_COLOR, 2)
            
            # Show original and resized dimensions
            cv2.putText(resized_frame, f"Original: {frame.shape[1]}x{frame.shape[0]}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(resized_frame, f"Resized: {TARGET_WIDTH}x{TARGET_HEIGHT}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Reference Line Visualization (720p)', resized_frame)
        
        # Keyboard controls
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('1'):
            Y1_LINE = max(0, Y1_LINE - 5)
        elif key == ord('2'):
            Y1_LINE = min(TARGET_HEIGHT, Y1_LINE + 5)
        elif key == ord('3'):
            Y2_LINE = max(0, Y2_LINE - 5)
        elif key == ord('4'):
            Y2_LINE = min(TARGET_HEIGHT, Y2_LINE + 5)
        elif key == ord('r'):  # Reset to default positions
            Y1_LINE = int(TARGET_HEIGHT * 0.4)
            Y2_LINE = int(TARGET_HEIGHT * 0.6)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Y1_LINE: {Y1_LINE}")
    print(f"Final Y2_LINE: {Y2_LINE}")
    print(f"Video dimensions: {TARGET_WIDTH}x{TARGET_HEIGHT}")

if __name__ == "__main__":
    import numpy as np
    visualize_reference_lines()