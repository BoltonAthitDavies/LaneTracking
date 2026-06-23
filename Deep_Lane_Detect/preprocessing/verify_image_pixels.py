"""
Author: Sippawit Thammawiset
Date: September 1, 2024.
File: verify_image_pixels.py
"""

from utils.dataset_loader import load_image_path_from_directory
import numpy as np
import cv2
import os
import argparse


def main() -> None:
    parse = argparse.ArgumentParser(description='Verify pixels of a batch of images.')
    parse.add_argument('-i', '--input', dest='input', type=str, required=True,
                       help='Directory containing the input images.')
    args = parse.parse_args()

    if not args.input:
        parse.print_help()
        return

    input_directory = args.input

    image_paths = load_image_path_from_directory(input_directory)

    images = []
    for image_path in image_paths:
        image_filename = os.path.split(image_path)[-1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        unique = np.unique(image)
        print(f'> {image_filename} has unique pixels: {unique}')
        images.append(image)
    images = np.array(images)

    print('[INFO] Unique pixels of a batch of images:', np.unique(images))


if __name__ == '__main__':
    main()
