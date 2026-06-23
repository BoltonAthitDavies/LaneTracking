"""
Author: Sippawit Thammawiset
Date: September 1, 2024.
File: find_hsv.py
"""

from utils import colors
from typing import Tuple
import numpy as np
import cv2
import argparse


def nothing(x):
    pass


SLIDER_WIN_NAME = 'Bars'
HL = 'H-Low'
HH = 'H-High'
SL = 'S-Low'
SH = 'S-High'
VL = 'V-Low'
VH = 'V-High'


def initialize_slider() -> None:
    cv2.namedWindow(SLIDER_WIN_NAME, flags=cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar(HL, SLIDER_WIN_NAME, 0, 179, nothing)
    cv2.createTrackbar(HH, SLIDER_WIN_NAME, 0, 179, nothing)
    cv2.createTrackbar(SL, SLIDER_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar(SH, SLIDER_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar(VL, SLIDER_WIN_NAME, 0, 255, nothing)
    cv2.createTrackbar(VH, SLIDER_WIN_NAME, 0, 255, nothing)

    # Set initial values for sliders
    cv2.setTrackbarPos(HL, SLIDER_WIN_NAME, 0)
    cv2.setTrackbarPos(HH, SLIDER_WIN_NAME, 179)
    cv2.setTrackbarPos(SL, SLIDER_WIN_NAME, 0)
    cv2.setTrackbarPos(SH, SLIDER_WIN_NAME, 255)
    cv2.setTrackbarPos(VL, SLIDER_WIN_NAME, 0)
    cv2.setTrackbarPos(VH, SLIDER_WIN_NAME, 255)


def find_hsv(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    hsv_low = np.array([0, 0, 0])
    hsv_high = np.array([179, 255, 255])

    try:
        while True:
            blur = cv2.GaussianBlur(image, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            h_low = cv2.getTrackbarPos(HL, SLIDER_WIN_NAME)
            h_high = cv2.getTrackbarPos(HH, SLIDER_WIN_NAME)
            s_low = cv2.getTrackbarPos(SL, SLIDER_WIN_NAME)
            s_high = cv2.getTrackbarPos(SH, SLIDER_WIN_NAME)
            v_low = cv2.getTrackbarPos(VL, SLIDER_WIN_NAME)
            v_high = cv2.getTrackbarPos(VH, SLIDER_WIN_NAME)

            hsv_low = np.array([h_low, s_low, v_low])
            hsv_high = np.array([h_high, s_high, v_high])

            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            cv2.imshow('Mask', mask)
            cv2.imshow('Preview', masked_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    finally:
        return hsv_low, hsv_high


def main() -> None:
    parse = argparse.ArgumentParser(description='Find HSV values using a slider.')
    parse.add_argument('-i', '--input', dest='input', type=str, required=True,
                       help='Path to the input image.')
    args = parse.parse_args()

    if not args.input:
        parse.print_help()
        return

    image_path: str = args.input

    initialize_slider()
    hsv_low, hsv_high = find_hsv(image_path)

    print('> HSV low:', hsv_low)
    print('> HSV high:', hsv_high)

    print(f'{colors.OKGREEN}[INFO] Done.{colors.ENDC}')


if __name__ == '__main__':
    main()
