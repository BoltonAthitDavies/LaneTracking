import cv2
import numpy as np
import json
import os

def create_tusimple_json_from_masks(
    mask_dir,
    original_image_dir,
    output_json_filename='label_data_generated.json',
    h_samples_step=10,
    lane_thickness_tolerance=5, # Max vertical distance to group points at a given y_sample
    min_lane_points=5 # Minimum number of valid (non -2) points for a lane to be included
):
    """
    Generates a TuSimple-like JSON annotation file from binary segmentation masks.

    This function processes binary masks (where lanes are white, background is black)
    to extract lane coordinates and formats them into a JSON file compatible with
    the TuSimple lane detection dataset structure.

    Args:
        mask_dir (str): Directory containing the binary lane segmentation masks.
                        Assumes mask filenames are the same as original image names
                        (e.g., '0.png', '1.png', etc.).
        original_image_dir (str): Directory containing the original raw images.
                                  These paths will be stored in the 'raw_file' field.
        output_json_filename (str): Name of the output JSON file. Each entry will be
                                    written on a new line, similar to TuSimple's format.
        h_samples_step (int): The vertical step size (in pixels) for sampling lane
                              x-coordinates. For example, 10 means samples at y=160, 170, 180...
        lane_thickness_tolerance (int): When sampling at a specific y-coordinate,
                                        this defines the vertical range (y_sample +/- tolerance)
                                        to consider points as part of the lane at that y-level.
                                        Useful for thicker lane lines in masks.
        min_lane_points (int): The minimum number of valid (non -2) x-coordinates a detected
                               lane must have across the h_samples to be considered a valid lane
                               and included in the output.
    """
    generated_data = []

    # Get all mask files from the specified directory
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not mask_files:
        print(f"Error: No image files found in '{mask_dir}'. Please check the directory and file extensions.")
        return

    print(f"Starting conversion for {len(mask_files)} mask files...")

    # --- Dynamically determine min_h_sample and max_h_sample ---
    overall_min_y = float('inf')
    image_height = None

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read mask '{mask_path}' for h_sample determination. Skipping.")
            continue

        if image_height is None:
            image_height = mask.shape[0] # Get height from the first valid mask

        # Find all white pixels (lane pixels)
        white_pixels_y = np.where(mask == 255)[0]

        if len(white_pixels_y) > 0:
            current_mask_min_y = np.min(white_pixels_y)
            if current_mask_min_y < overall_min_y:
                overall_min_y = current_mask_min_y
    
    # If no white pixels were found in any mask, default to a reasonable min_h_sample
    if overall_min_y == float('inf'):
        print("Warning: No white lane pixels found in any mask. Defaulting min_h_sample to 0.")
        min_h_sample = 0
    else:
        min_h_sample = int(overall_min_y)

    # Set max_h_sample to the bottom of the image if height was determined
    if image_height is None:
        print("Error: Could not determine image height from any mask. Defaulting max_h_sample to 780.")
        max_h_sample = 780 # Fallback if no masks could be read
    else:
        max_h_sample = image_height - 1 # Use the actual image height

    # Define the list of y-coordinates (h_samples) where lane points will be sampled
    # Ensure h_samples are within image bounds
    h_samples = list(range(min_h_sample, max_h_sample + 1, h_samples_step))
    h_samples = [y for y in h_samples if y >= 0 and y < image_height] # Filter out-of-bounds samples


    print(f"Dynamically determined h_samples range: min_h_sample={min_h_sample}, max_h_sample={max_h_sample}")
    print(f"Generated h_samples: {h_samples}")


    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)

        # Infer the original image filename from the mask filename.
        # Since raw and mask images have the same name, the mask_file itself is the base.
        original_image_base_with_ext = mask_file # e.g., '0.png'

        # Construct the full path to the raw image
        raw_file_path = os.path.join(original_image_dir, original_image_base_with_ext)

        if not os.path.exists(raw_file_path):
            print(f"Warning: Original image '{raw_file_path}' not found. Skipping mask '{mask_file}'.")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read mask '{mask_path}'. Skipping.")
            continue

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        image_lanes = []

        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            lane_x_coords = [-2] * len(h_samples)

            for i, y_sample in enumerate(h_samples):
                # Corrected: Access y-coordinate using p[0][1]
                points_at_y = [
                    p[0] for p in contour
                    if p[0][1] >= y_sample - lane_thickness_tolerance and p[0][1] <= y_sample + lane_thickness_tolerance
                ]

                if points_at_y:
                    avg_x = int(np.mean([p[0] for p in points_at_y]))
                    lane_x_coords[i] = avg_x

            valid_points_count = sum(1 for x in lane_x_coords if x != -2)
            if valid_points_count >= min_lane_points:
                image_lanes.append(lane_x_coords)

        def get_sort_key(lane):
            for i, x in reversed(list(enumerate(lane))):
                if x != -2:
                    return x
            return float('inf')

        image_lanes.sort(key=get_sort_key)

        generated_data.append({
            "lanes": image_lanes,
            "raw_file": raw_file_path,
            "h_samples": h_samples
        })

        print(f"Processed mask: '{mask_file}'. Found {len(image_lanes)} lanes.")

    with open(output_json_filename, 'w') as f:
        for entry in generated_data:
            f.write(json.dumps(entry) + '\n')

    print(f"\nSuccessfully generated '{output_json_filename}' with data for {len(generated_data)} images.")
    print(f"Final h_samples used for generation: {h_samples}")

# --- How to Use This Code ---
# 1. Save the code above as a Python file (e.g., `mask_to_tusimple.py`).
# 2. Organize your files:
#    - Create a directory for your binary segmentation masks (e.g., `my_masks/`).
#      Make sure your mask files are named consistently (e.g., `0.png`).
#    - Create a directory for your original raw images (e.g., `my_raw_images/`).
#      Make sure original images correspond to masks and have the same names (e.g., `0.png`).
# 3. Call the function with your directory paths:

# Example usage (uncomment and modify paths to run):
if __name__ == '__main__':
    # IMPORTANT: Replace 'path/to/your/mask_directory' and 'path/to/your/original_images_directory'
    # with the actual paths on your system.
    create_tusimple_json_from_masks(
        mask_dir='dataset/high_brightness_1/bin_masks/img',
        original_image_dir='dataset/high_brightness_1/images/img',
        output_json_filename='label_data_high_brightness_1.json',
        h_samples_step=10,
        lane_thickness_tolerance=5,
        min_lane_points=5
    )

    print("\nRemember to uncomment the example usage block and update paths to run the script!")
