"""
Author: Natthaphat + Sippawit + ChatGPT
Date: July 2025
File: generate_tusimple_from_original.py
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from skimage.morphology import dilation
from PIL import Image, ImageFilter

# Lane HSV thresholds (for purple lane as example)
H_LOW, S_LOW, V_LOW = 0, 45, 100
H_HIGH, S_HIGH, V_HIGH = 179, 255, 255

def preprocessing(image: np.ndarray) -> np.ndarray:
    dilated_mask = dilation(image, np.ones((7, 7), np.uint8))
    pil_image = Image.fromarray(dilated_mask)
    filtered_image = pil_image.filter(ImageFilter.ModeFilter(13))
    return np.array(filtered_image)

def extract_lanes_from_image(image: np.ndarray,
                             hsv_low: np.ndarray,
                             hsv_high: np.ndarray,
                             h_samples: list[int],
                             thickness_tolerance: int = 5,
                             min_lane_points: int = 5) -> list:
    """
    Return lane list like TuSimple format: [ [x1, x2, ..., xn], ... ]
    """
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
    lane_mask = cv2.inRange(hsv_image, hsv_low, hsv_high)

    roi_mask = np.zeros_like(lane_mask)
    height = image.shape[0]
    roi_mask[height // 2:-1, ...] = 255
    lane_mask = cv2.bitwise_and(lane_mask, roi_mask)

    binary_mask = preprocessing(lane_mask)

    # Extract contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lanes = []

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue

        lane = [-2] * len(h_samples)
        for i, y_sample in enumerate(h_samples):
            points_at_y = [p[0] for p in contour
                           if y_sample - thickness_tolerance <= p[0][1] <= y_sample + thickness_tolerance]
            if points_at_y:
                avg_x = int(np.mean([p[0] for p in points_at_y]))
                lane[i] = avg_x

        if sum(x != -2 for x in lane) >= min_lane_points:
            lanes.append(lane)

    # Sort lanes left to right
    def lane_sort_key(lane):
        for x in reversed(lane):
            if x != -2:
                return x
        return float('inf')

    lanes.sort(key=lane_sort_key)
    return lanes

def generate_tusimple_from_images(
        image_dir: str,
        output_json_path: str,
        h_samples_step: int = 10,
        min_h_sample: int = 160,
        lane_thickness_tolerance: int = 5,
        min_lane_points: int = 5
    ):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f'[ERROR] No image files found in "{image_dir}".')
        return

    # Read image height from first valid image
    sample_img = cv2.imread(os.path.join(image_dir, image_files[0]))
    if sample_img is None:
        print('[ERROR] Cannot read any image.')
        return
    image_height = sample_img.shape[0]
    h_samples = list(range(min_h_sample, image_height - 1, h_samples_step))

    hsv_low = np.array([H_LOW, S_LOW, V_LOW])
    hsv_high = np.array([H_HIGH, S_HIGH, V_HIGH])
    tusimple_data = []

    for filename in tqdm(image_files, desc='Generating TuSimple'):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARNING] Skipped unreadable image: {filename}')
            continue

        lanes = extract_lanes_from_image(img, hsv_low, hsv_high, h_samples,
                                         thickness_tolerance=lane_thickness_tolerance,
                                         min_lane_points=min_lane_points)

        if not lanes:
            print(f'[WARNING] No valid lanes in {filename}')
            continue

        tusimple_data.append({
            "lanes": lanes,
            "h_samples": h_samples,
            "raw_file": os.path.join(image_dir, filename)
        })

    # Save
    with open(output_json_path, 'w') as f:
        for entry in tusimple_data:
            json.dump(entry, f)
            f.write('\n')

    print(f'[INFO] Saved TuSimple-style JSON to {output_json_path} with {len(tusimple_data)} entries.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate TuSimple JSON directly from images.')
    parser.add_argument('-i', '--image-dir', type=str, required=True,
                        help='Path to directory with raw images.')
    parser.add_argument('-o', '--output-json', type=str, default='tusimple_generated.json',
                        help='Path to output JSON file.')
    args = parser.parse_args()

    generate_tusimple_from_images(
        image_dir=args.image_dir,
        output_json_path=args.output_json,
        h_samples_step=10,
        min_h_sample=160,
        lane_thickness_tolerance=5,
        min_lane_points=5
    )

# python preprocessing/generate_tusimple_from_original.py --image-dir dataset/test/images/img --output-json test_label.json
# python preprocessing/generate_tusimple_from_original.py --image-dir dataset/training/images/img --output-json label_data_training.json