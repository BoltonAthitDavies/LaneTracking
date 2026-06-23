"""
Author: Sippawit Thammawiset
Date: September 1, 2024.
File: split_train_test.py
"""

from sklearn.model_selection import train_test_split
from typing import Tuple, List
from utils import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse

OUTPUT_TRAIN_IMAGES = 'training/images/img'
OUTPUT_TRAIN_BIN_MASKS = 'training/bin_masks/img'
OUTPUT_TRAIN_INST_MASKS = 'training/inst_masks/img'
OUTPUT_TEST_IMAGES = 'test/images/img'
OUTPUT_TEST_BIN_MASKS = 'test/bin_masks/img'
OUTPUT_TEST_INST_MASKS = 'test/inst_masks/img'


def load_images_masks(input_directory: str) -> Tuple[List[str], List[str], List[str]]:
    images = []
    bin_masks = []
    inst_masks = []

    for subdir in sorted(os.listdir(input_directory)):
        subdir_path = os.path.join(input_directory, subdir)
        if os.path.isdir(subdir_path):
            image_dir = os.path.join(subdir_path, 'images', 'img')
            bin_mask_dir = os.path.join(subdir_path, 'bin_masks', 'img')
            inst_mask_dir = os.path.join(subdir_path, 'inst_masks', 'img')

            if os.path.exists(image_dir) and os.path.exists(bin_mask_dir) and os.path.exists(inst_mask_dir):
                image_files = sorted(os.listdir(image_dir))
                for image_file in image_files:
                    if image_file.endswith('.png'):
                        image_path = os.path.join(image_dir, image_file)
                        bin_mask_path = os.path.join(bin_mask_dir, 'bin_mask-' + image_file)
                        inst_mask_path = os.path.join(inst_mask_dir, 'inst_mask-' + image_file)

                        # Ensure corresponding mask files exist.
                        if os.path.exists(bin_mask_path) and os.path.exists(inst_mask_path):
                            images.append(image_path)
                            bin_masks.append(bin_mask_path)
                            inst_masks.append(inst_mask_path)
                        else:
                            print(f'{colors.WARNING}'
                                  f'[WARNING] Mask files not found for {image_file} in {subdir}. Skipped.'
                                  f'{colors.ENDC}')
            else:
                print(f'{colors.WARNING}'
                      f'[WARNING] Directories not found in {subdir}. Skipped.'
                      f'{colors.ENDC}')
        else:
            ...

    return images, bin_masks, inst_masks


def visualize(images: np.ndarray[object],
              bin_masks: np.ndarray[object],
              inst_masks: np.ndarray[object]) -> None:
    figure, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    random_indices = np.random.choice(range(len(images)), 3, replace=False)

    for i in range(3):
        image = cv2.imread(images[random_indices[i]])
        bin_mask = cv2.imread(bin_masks[random_indices[i]])
        inst_mask = cv2.imread(inst_masks[random_indices[i]])

        axes[i][0].imshow(image)
        axes[i][1].imshow(bin_mask, cmap='gray')
        axes[i][2].imshow(inst_mask, cmap='gray')

    axes[0][0].set_title('Image')
    axes[0][1].set_title('Binary Mask')
    axes[0][2].set_title('Instance Mask')

    plt.tight_layout()
    plt.show()


def save(image_paths: np.ndarray,
         input_dir: str,
         output_dir: str) -> None:
    i = 0
    for image_path in image_paths:
        filename = f'{i}.jpg' # Change .png or .jpg save in train, test
        image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(input_dir, output_dir, filename), image)
        i += 1


def main() -> None:
    parse = argparse.ArgumentParser(description='Split preprocessed dataset into training and test sets.')
    parse.add_argument('-i', '--input-directory', dest='input_directory', type=str, required=True,
                       help='Directory containing the preprocessed dataset.')
    parse.add_argument('-ts', '--test-size', dest='test_size', type=float, required=False, default=0.3,
                       help='Proportion of the dataset to include in the test split. Default is 0.3.')
    parse.add_argument('-sh', '--shuffle', dest='shuffle', action='store_true',
                       help='Flag to shuffle the dataset before splitting. Default is False.')
    parse.add_argument('-rs', '--random-state', dest='random_state', type=int, required=False, default=42,
                       help='Seed for the random number generator to ensure reproducibility. Default is 42.')
    args = parse.parse_args()

    if not args.input_directory:
        parse.print_help()
        return

    input_directory: str = args.input_directory
    test_size: float = args.test_size
    shuffle: bool = args.shuffle
    random_state: int = args.random_state

    os.makedirs(os.path.join(input_directory, OUTPUT_TRAIN_IMAGES), exist_ok=True)
    os.makedirs(os.path.join(input_directory, OUTPUT_TRAIN_BIN_MASKS), exist_ok=True)
    os.makedirs(os.path.join(input_directory, OUTPUT_TRAIN_INST_MASKS), exist_ok=True)
    os.makedirs(os.path.join(input_directory, OUTPUT_TEST_IMAGES), exist_ok=True)
    os.makedirs(os.path.join(input_directory, OUTPUT_TEST_BIN_MASKS), exist_ok=True)
    os.makedirs(os.path.join(input_directory, OUTPUT_TEST_INST_MASKS), exist_ok=True)

    images, bin_masks, inst_masks = load_images_masks(input_directory)

    images: np.ndarray = np.array(images)
    bin_masks: np.ndarray = np.array(bin_masks)
    inst_masks: np.ndarray = np.array(inst_masks)

    print('[INFO] Splitting the dataset into training and test sets. This may take some time...')
    train_images, test_images, train_bin_masks, test_bin_masks, train_inst_masks, test_inst_masks = train_test_split(
        images, bin_masks, inst_masks, test_size=test_size, shuffle=shuffle, random_state=random_state)

    print('> Training set shape:', train_images.shape)
    print('> Test set shape:', test_images.shape)

    visualize(train_images, train_bin_masks, train_inst_masks)

    print('[INFO] Saving into "training/" and "test/". This may take some time...')
    save(train_images, input_directory, OUTPUT_TRAIN_IMAGES)
    save(train_bin_masks, input_directory, OUTPUT_TRAIN_BIN_MASKS)
    save(train_inst_masks, input_directory, OUTPUT_TRAIN_INST_MASKS)

    save(test_images, input_directory, OUTPUT_TEST_IMAGES)
    save(test_bin_masks, input_directory, OUTPUT_TEST_BIN_MASKS)
    save(test_inst_masks, input_directory, OUTPUT_TEST_INST_MASKS)

    print(f'{colors.OKGREEN}[INFO] Saved training dataset to "{input_directory}/training/".{colors.ENDC}')
    print(f'{colors.OKGREEN}[INFO] Saved test dataset to "{input_directory}/test/".{colors.ENDC}')
    print(f'{colors.OKGREEN}[INFO] Done.{colors.ENDC}')


if __name__ == '__main__':
    main()
