import cv2
import numpy as np
import os
from tqdm import tqdm

# --- 1. Configuration ---

# Set your input and output video files
INPUT_FILE = '/home/nvidia/LaneTracking/Deep_Lane_Detect/raw_dataset/high_bright_25_ccw.mp4'
OUTPUT_FILE = '/home/nvidia/LaneTracking/Deep_Lane_Detect/raw_dataset/test_bright5.mp4'

# Set the brightness adjustment value:
#  Positive (e.g., 50): Increases brightness
#  Negative (e.g., -50): Decreases brightness
#  Zero (0): No change
BRIGHTNESS_ADJUST = 180

# --- 2. Brightness Adjustment Function ---

def adjust_brightness_hsv(image, value):
    """
    Adjusts the brightness of an image using the HSV color space.
    """
    if value == 0:
        return image

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Use np.int16 to allow for temporary values > 255 or < 0
    v_new = v.astype(np.int16) + value
    
    # Clip the values to stay in the valid 0-255 range
    v_new = np.clip(v_new, 0, 255).astype(np.uint8)

    # Merge channels back
    final_hsv = cv2.merge([h, s, v_new])
    
    # Convert back to BGR
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_image

# --- 3. Main Video Processing Function ---

def process_video(input_path, output_path, brightness_value):
    # Open the input video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties (width, height, fps, total frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {input_path}")
    print(f"  -> Properties: {width}x{height} @ {fps:.2f} FPS")
    print(f"  -> Total frames: {total_frames}")
    print(f"Output video: {output_path}")
    print(f"Brightness adjustment: {brightness_value}")

    # Define the codec and create VideoWriter object
    # 'mp4v' is a common codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video file {output_path}")
        cap.release()
        return

    # Process video frame by frame with a progress bar
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        
        # If frame read failed (e.g., end of video), break the loop
        if not ret:
            break
            
        # Apply the brightness adjustment
        bright_frame = adjust_brightness_hsv(frame, brightness_value)
        
        # Write the adjusted frame to the output video
        out.write(bright_frame)

    # Release video capture and writer objects
    print("\nProcessing complete.")
    cap.release()
    out.release()
    print(f"Video saved successfully to {output_path}")

# --- 4. Run the script ---

if __name__ == "__main__":
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
    else:
        process_video(INPUT_FILE, OUTPUT_FILE, BRIGHTNESS_ADJUST)