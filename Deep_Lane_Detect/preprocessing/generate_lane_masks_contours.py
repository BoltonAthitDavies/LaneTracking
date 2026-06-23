"""
Author: Sippawit Thammawiset
Date: September 8, 2024.
File: generate_lane_masks_contours.py
"""

from skimage.morphology import dilation
from PIL import Image, ImageFilter
from typing import Union, Tuple
from tqdm import tqdm
from utils import colors, debug
from utils.image_path_loader import load_image_path_from_directory
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse

# H_LOW = 0
# S_LOW = 96
# V_LOW = 103

# For purple lane
H_LOW = 0
S_LOW = 45
V_LOW = 100
H_HIGH = 179
S_HIGH = 255
V_HIGH = 255


def preprocessing(image: np.ndarray) -> np.ndarray:
    dilated_mask = dilation(image, np.ones((7, 7), np.uint8))

    pil_image = Image.fromarray(dilated_mask)
    filtered_image = pil_image.filter(ImageFilter.ModeFilter(13))
    output = np.array(filtered_image)

    return output


def create_lane_mask(input_directory: str,
                     color_mode: str,
                     left_lane_mask_color: Union[int, Tuple[int, int, int]],
                     right_lane_mask_color: Union[int, Tuple[int, int, int]],
                     target_size: Union[Tuple[int, int], Tuple[None, None]]) -> None:
    """
    Creates semantic and instance masks for left and right lanes from input images.

    This function processes images located in the "images/img" subdirectory of the `input_directory`
    to generate semantic and instance masks highlighting the left and right lane markings in each image.
    The masks are saved in "bin_masks/img" and "inst_masks/img" subdirectories, respectively.

    Args:
        input_directory (str): Directory containing the input images. The function expects images to be located
            in the "images/img" subdirectory of the `input_directory`.
        color_mode (str): Color mode for the output instance masks. Can be "grayscale" or "rgb".
            - "grayscale": The instance mask will be a single-channel image.
            - "rgb": The instance mask will be a three-channel image with specified colors.
        left_lane_mask_color (Union[int, Tuple[int, int, int]]): The color value to assign to the left lane in the instance mask.
            - If "color_mode" is "grayscale", this should be an integer between 0 and 255.
            - If "color_mode" is "rgb", this should be a tuple of three integers representing
            (R, G, B) color values, each between 0 and 255.
        right_lane_mask_color (Union[int, Tuple[int, int, int]]): The color value to assign to the right lane in the instance mask.
            Same format as `left_lane_mask_color`.
        target_size (Union[Tuple[int, int], Tuple[None, None]]): The desired size `(width, height)` of the output masks.
            If `(None, None)`, the original image size is used.

    Returns:
        None

    Notes:
        - The function uses predefined HSV color thresholds (`H_LOW`, `S_LOW`, `V_LOW`, `H_HIGH`,
          `S_HIGH`, `V_HIGH`) for lane detection in HSV color space.

        Implemented by Sippawit Thammawiset.
    """

    input_image_dir = os.path.join(input_directory, 'images', 'img')
    output_bin_mask_dir = os.path.join(input_directory, 'bin_masks', 'img')
    output_inst_mask_dir = os.path.join(input_directory, 'inst_masks', 'img')

    os.makedirs(output_bin_mask_dir, exist_ok=True)
    os.makedirs(output_inst_mask_dir, exist_ok=True)

    image_filepaths = load_image_path_from_directory(input_image_dir)
    n = len(image_filepaths)

    if n == 0:
        raise AssertionError(
            f'No images found in "{input_image_dir}/".'
        )

    hsv_low = np.array([H_LOW, S_LOW, V_LOW])
    hsv_high = np.array([H_HIGH, S_HIGH, V_HIGH])

    for i in tqdm(range(n), desc='Masking lanes', unit=' image'):
        filename_with_ext = image_filepaths[i].split('/')[-1]

        image = cv2.imread(image_filepaths[i])
        height, width, channels = image.shape

        blur_image = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

        lane_mask = cv2.inRange(hsv_image, hsv_low, hsv_high)

        roi_mask = np.zeros_like(lane_mask)
        roi_mask[height // 2:-1, ...] = 255

        lane_mask = cv2.bitwise_and(lane_mask, roi_mask)

        preprocessed = preprocessing(lane_mask)  # shape: (height, width)

        contours, hierarchy = cv2.findContours(preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lane_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # contour = cv2.drawContours(image, lane_contours[:2], -1, (0, 255, 0), 10)
        # debug.plot(contour)

        # If this condition is passed, there is at least an element in a list.
        if len(lane_contours) == 0:
            print(f'{colors.WARNING}'
                  f'[WARNING] No lane contours found in "{image_filepaths[i]}". Skipping.'
                  f'{colors.ENDC}')
            continue

        M1 = cv2.moments(lane_contours[0])
        cx1 = int(M1['m10'] / (M1['m00'] + 0.0001))

        cx2 = width // 2
        found_one_lane = False

        try:
            M2 = cv2.moments(lane_contours[1])
            cx2 = int(M2['m10'] / (M2['m00'] + 0.0001))
        except IndexError:
            print(f'{colors.WARNING}'
                  f'[WARNING] Only one lane contour is found in "{image_filepaths[i]}".'
                  f'{colors.ENDC}')
            found_one_lane = True

        left_lane_mask = np.zeros_like(preprocessed, dtype=np.uint8)  # shape: (height, width)
        right_lane_mask = np.zeros_like(preprocessed, dtype=np.uint8)  # shape: (height, width)

        if found_one_lane:
            if cx1 < cx2:
                left_lane_contour = lane_contours[0]
                cv2.fillPoly(left_lane_mask, np.array([left_lane_contour], dtype=np.int32), (255, 255, 255))
            else:
                right_lane_contour = lane_contours[0]
                cv2.fillPoly(right_lane_mask, np.array([right_lane_contour], dtype=np.int32), (255, 255, 255))
        else:
            left_lane_contour = lane_contours[0] if cx1 < cx2 else lane_contours[1]
            right_lane_contour = lane_contours[1] if cx1 < cx2 else lane_contours[0]
            cv2.fillPoly(left_lane_mask, np.array([left_lane_contour], dtype=np.int32), (255, 255, 255))
            cv2.fillPoly(right_lane_mask, np.array([right_lane_contour], dtype=np.int32), (255, 255, 255))

        if target_size == (None, None):
            target_size = (width, height)

        left_lane_mask = cv2.resize(left_lane_mask, target_size, interpolation=cv2.INTER_LINEAR)
        right_lane_mask = cv2.resize(right_lane_mask, target_size, interpolation=cv2.INTER_LINEAR)

        # semantic mask
        left_lane_bin_mask = left_lane_mask.copy()
        left_lane_bin_mask[left_lane_bin_mask.nonzero()] = 255
        right_lane_bin_mask = right_lane_mask.copy()
        right_lane_bin_mask[right_lane_bin_mask.nonzero()] = 255

        # Instance mask
        left_lane_inst_mask = left_lane_mask.copy()
        right_lane_inst_mask = right_lane_mask.copy()

        if color_mode == 'grayscale':
            left_lane_inst_mask[left_lane_inst_mask.nonzero()] = left_lane_mask_color
            right_lane_inst_mask[right_lane_inst_mask.nonzero()] = right_lane_mask_color
        elif color_mode == 'rgb':
            left_lane_inst_mask = np.stack((left_lane_inst_mask,) * 3, axis=-1)
            right_lane_inst_mask = np.stack((right_lane_inst_mask,)*3, axis=-1)

            mask_coords = left_lane_inst_mask.nonzero()
            for i in range(len(mask_coords[0])):
                x, y = mask_coords[0][i], mask_coords[1][i]
                left_lane_inst_mask[x, y, 0] = left_lane_mask_color[0]
                left_lane_inst_mask[x, y, 1] = left_lane_mask_color[1]
                left_lane_inst_mask[x, y, 2] = left_lane_mask_color[2]

            mask_coords = right_lane_inst_mask.nonzero()
            for i in range(len(mask_coords[0])):
                x, y = mask_coords[0][i], mask_coords[1][i]
                right_lane_inst_mask[x, y, 0] = right_lane_mask_color[0]
                right_lane_inst_mask[x, y, 1] = right_lane_mask_color[1]
                right_lane_inst_mask[x, y, 2] = right_lane_mask_color[2]

        combined_bin_mask = np.add(left_lane_bin_mask, right_lane_bin_mask)
        combined_inst_mask = cv2.add(left_lane_inst_mask, right_lane_inst_mask)

        filtered_image = Image.fromarray(combined_bin_mask).filter(ImageFilter.ModeFilter(13))
        bin_mask = np.array(filtered_image)
        filtered_image = Image.fromarray(combined_inst_mask).filter(ImageFilter.ModeFilter(13))
        inst_mask = np.array(filtered_image)

        # debug.plot(inst_mask)

        cv2.imwrite(os.path.join(output_bin_mask_dir, 'bin_mask-' + filename_with_ext), bin_mask)
        cv2.imwrite(os.path.join(output_inst_mask_dir, 'inst_mask-' + filename_with_ext), inst_mask)


def main() -> None:
    parse = argparse.ArgumentParser(description='Create lane masks using contours.')
    parse.add_argument('-i', '--input-directory', dest='input_directory', type=str, required=True,
                       help='Directory containing the "images/img" subdirectory.')
    parse.add_argument('-c', '--color-mode', dest='color_mode', type=str, default='grayscale', required=False,
                       help='Color mode for saving mask images: "rgb" or "grayscale". Default is "grayscale".')
    parse.add_argument('-lc', '--left-mask-color', dest='left_lane_mask_color', type=str, default='100', required=False,
                       help='Color value for the left lane mask. If "color-mode" is set to "rgb", '
                            'provide the value as "R,G,B". Otherwise, provide a single 8-bit value. Default is "100".')
    parse.add_argument('-rc', '--right-mask-color', dest='right_lane_mask_color', type=str, default='200', required=False,
                       help='Color value for the right lane mask. If "color-mode" is set to "rgb", '
                            'provide the value as "R,G,B". Otherwise, provide a single 8-bit value. Default is "200".')
    parse.add_argument('-tw', '--target-width', dest='target_width', type=int, required=False,
                       help='Target width for resizing the mask images.')
    parse.add_argument('-th', '--target-height', dest='target_height', type=int, required=False,
                       help='Target height for resizing the mask images.')
    args = parse.parse_args()

    if not args.input_directory:
        parse.print_help()
        return

    input_directory: str = args.input_directory
    color_mode: str = args.color_mode
    if color_mode == 'rgb':
        if args.left_lane_mask_color == '100':
            left_lane_mask_color = (100, 100, 100)
        else:
            left_mask_rgb: list[str] = args.left_lane_mask_color.split(',')

            if len(left_mask_rgb) != 3:
                raise ValueError(
                    'When using "color-mode" as "rgb", '
                    'the "left-mask-color" must be specified in the format "R,G,B". '
                    f'Receive: left-mask-color=({args.left_lane_mask_color})'
                )

            try:
                left_lane_mask_color = tuple([int(left_mask_rgb[0]),
                                              int(left_mask_rgb[1]),
                                              int(left_mask_rgb[2])])
            except ValueError:
                raise ValueError(
                    'Invalid "R,G,B". '
                    f'Receive: left-mask-color=({args.left_lane_mask_color})'
                )

        if args.right_lane_mask_color == '200':
            right_lane_mask_color = (200, 200, 200)
        else:
            right_mask_rgb: list[str] = args.right_lane_mask_color.split(',')

            if len(right_mask_rgb) != 3:
                raise ValueError(
                    'When using "color-mode" as "rgb", '
                    'the "right-mask-color" must be specified in the format "R,G,B". '
                    f'Receive: right-mask-color=({args.right_lane_mask_color})'
                )

            try:
                right_lane_mask_color = tuple([int(right_mask_rgb[0]),
                                               int(right_mask_rgb[1]),
                                               int(right_mask_rgb[2])])
            except ValueError:
                raise ValueError(
                    'Invalid "R,G,B". '
                    f'Receive: right-mask-color=({args.right_lane_mask_color})'
                )
    elif color_mode == 'grayscale':
        try:
            left_lane_mask_color = int(args.left_lane_mask_color)
        except ValueError:
            raise ValueError(
                'Invalid 8-bit color value. '
                f'Receive: left-mask-color={args.left_lane_mask_color}'
            )

        try:
            right_lane_mask_color = int(args.right_lane_mask_color)
        except ValueError:
            raise ValueError(
                'Invalid 8-bit color value. '
                f'Receive: right-mask-color={args.right_lane_mask_color}'
            )
    else:
        raise ValueError(
            'Invalid "color_mode" argument. Expected "rgb" or "grayscale". '
            f'Receive: {args.color_mode}'
        )

    target_width = args.target_width
    target_height = args.target_height

    if (target_width is not None and target_width < 0) or \
       (target_height is not None and target_height < 0):
        raise ValueError(
            'Target width and height must be non-negative integers. '
            f'Received: target-width={target_width}, target-height={target_height}'
        )

    create_lane_mask(input_directory,
                     color_mode,
                     left_lane_mask_color,
                     right_lane_mask_color,
                     (target_width, target_height))

    print(f'{colors.OKGREEN}[INFO] Saved semantic masks to "{input_directory}/bin_masks/img".{colors.ENDC}')
    print(f'{colors.OKGREEN}[INFO] Saved instance masks to "{input_directory}/inst_masks/img".{colors.ENDC}')
    print(f'{colors.OKGREEN}[INFO] Done.{colors.ENDC}')


if __name__ == '__main__':
    main()
