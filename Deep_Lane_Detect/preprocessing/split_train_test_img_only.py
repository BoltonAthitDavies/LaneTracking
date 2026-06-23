"""
Author: Sippawit + ChatGPT
Date: Updated July 2025.
File: split_train_test_images_recursive.py
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

OUTPUT_TRAIN_IMAGES = 'training/images/img'
OUTPUT_TEST_IMAGES = 'test/images/img'

def find_all_image_paths(root_dir: str) -> list[str]:
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == 'img' and os.path.basename(os.path.dirname(dirpath)) == 'images':
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(dirpath, f))
    print(image_paths)
    return sorted(image_paths)

def save_images(image_paths: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, path in enumerate(tqdm(image_paths, desc=f"Saving to {output_dir}")):
        img = cv2.imread(path)
        if img is not None:
            filename = f"{i:04d}.jpg"  # or .png if you prefer
            cv2.imwrite(os.path.join(output_dir, filename), img)
        else:
            print(f"[WARNING] Could not read {path}. Skipped.")

def main():
    parser = argparse.ArgumentParser(description='Recursively split image dataset into training and test sets.')
    parser.add_argument('-i', '--input-directory', required=True,
                        help='Root directory containing subfolders with "images/img".')
    parser.add_argument('-ts', '--test-size', type=float, default=0.3,
                        help='Proportion of dataset for test set. Default = 0.3')
    parser.add_argument('-sh', '--shuffle', action='store_true',
                        help='Shuffle dataset before splitting.')
    parser.add_argument('-rs', '--random-state', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    image_paths = find_all_image_paths(args.input_directory)

    if not image_paths:
        print(f"[ERROR] No image files found under '{args.input_directory}' in any 'images/img' subfolder.")
        return

    print(f"[INFO] Found {len(image_paths)} images across all subfolders.")

    train_imgs, test_imgs = train_test_split(
        image_paths,
        test_size=args.test_size,
        shuffle=args.shuffle,
        random_state=args.random_state
    )

    save_images(train_imgs, os.path.join(args.input_directory, OUTPUT_TRAIN_IMAGES))
    save_images(test_imgs, os.path.join(args.input_directory, OUTPUT_TEST_IMAGES))

    print(f"[INFO] Training set: {len(train_imgs)} images")
    print(f"[INFO] Test set: {len(test_imgs)} images")
    print(f"[INFO] Output saved to:")
    print(f"    {os.path.join(args.input_directory, OUTPUT_TRAIN_IMAGES)}")
    print(f"    {os.path.join(args.input_directory, OUTPUT_TEST_IMAGES)}")

if __name__ == '__main__':
    main()
