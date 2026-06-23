"""
Author: Sippawit Thammawiset
Date: September 1, 2024.
File: find_ipm.py
"""

from skimage.morphology import thin
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

# Camera
CAMERA_MATRIX = np.array([
    [204.012775, 0.000000, 211.861709],
    [0.000000, 204.594692, 123.687077],
    [0.000000, 0.000000, 1.000000],
])
DIST_COEFFS = np.array([-0.060503, 0.031436, -0.001866, 0.001043, 0.000000])

# HSV
H_LOW = 0
H_HIGH = 179
S_LOW = 70
S_HIGH = 255
V_LOW = 115
V_HIGH = 255

# ROI
"""
ROI vertices:
          Middle
          /   \
Lower Left --- Lower Right
"""
ROI_VERTICES_MIDDLE_OFFSET_X = -25
ROI_VERTICES_MIDDLE_OFFSET_Y = -80

# Trapezoid
TOP_OFFSET = 40
BOTTOM_OFFSET = 0
BOTTOM_WIDTH = 500


def show_image(image: np.ndarray,
               title: str,
               cmap: Optional[str] = None):
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.show()


def get_vanishing_point(lines: np.ndarray) -> np.ndarray:
    """
    Computes the vanishing point from a set of lines by solving the least-squares problem.

    Args:
        lines (np.ndarray): Array of lines, where each line is represented by its start and end points as
                            [[begin_x, begin_y, end_x, end_y], ...]. The shape is (N, 1, 4), where N is the number of lines.

    Returns:
        np.ndarray: The computed vanishing point as a 2D point [u, v].

    Notes:
        The function computes the vanishing point by finding the intersection of lines in a least-squares sense.
        The normal vector to each line segment is calculated, and the vanishing point is determined by solving
        a linear system based on these normals.
    """

    # Initialize matrices for the least-squares calculation
    Lhs = np.zeros((2, 2), dtype=np.float32)  # Left-hand side matrix
    Rhs = np.zeros((2, 1), dtype=np.float32)  # Right-hand side matrix

    for line in lines:
        for begin_x, begin_y, end_x, end_y in line:
            # Compute the normal vector to the line segment
            normal = np.array([[-(end_y - begin_y)], [end_x - begin_x]], dtype=np.float32)

            # Normalize the normal vector
            normal /= np.linalg.norm(normal)

            # Convert the starting point of the line segment to a column vector
            point = np.array([[begin_x], [begin_y]], dtype=np.float32)

            # Compute the outer product of the normal vector with itself
            outer = np.matmul(normal, normal.T)

            # Accumulate the outer product into the left-hand side matrix
            Lhs += outer

            # Accumulate the result of the outer product and the point into the right-hand side matrix
            Rhs += np.matmul(outer, point)

    # Solve the linear system to get the vanishing point
    vanishing_point = np.matmul(np.linalg.inv(Lhs), Rhs)

    return vanishing_point.ravel()


def on_line(p1: np.ndarray, p2: np.ndarray, y: float) -> np.ndarray:
    return np.array([p1[0] + (p2[0] - p1[0]) / float(p2[1] - p1[1]) * (y - p1[1]), y])


def get_roi(target_size: Tuple[int, int],):
    roi_vertices = np.array([
        [0, target_size[1]],
        [target_size[0], target_size[1]],
        [target_size[0] // 2 + ROI_VERTICES_MIDDLE_OFFSET_X,
         target_size[1] // 2 + ROI_VERTICES_MIDDLE_OFFSET_Y]
    ], dtype=np.int32)
    roi = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    cv2.fillPoly(roi, [roi_vertices], (255, 255, 255))

    return roi


def main() -> None:
    parse = argparse.ArgumentParser(description='Find Inverse Perspective Mapping (IPM).')
    parse.add_argument('-i', '--input', dest='input', type=str, required=True,
                       help='Path to the input image.')

    args = parse.parse_args()

    if not args.input:
        parse.print_help()
        return

    image_path: str = args.input

    image = cv2.imread(image_path)
    height, width, channels = image.shape
    resize_image = cv2.resize(image, (width, height))
    rgb_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    show_image(rgb_image, 'Original Image')

    # Step 1: Undistort the image using the camera matrix and distortion coefficients.
    undistorted_image = cv2.undistort(rgb_image, CAMERA_MATRIX, DIST_COEFFS)
    show_image(undistorted_image, 'Undistorted Image')

    # Step 2: Apply Gaussian blur and convert the image to HSV color space.
    blur_image = cv2.GaussianBlur(undistorted_image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_RGB2HSV)

    # Create a binary mask using the HSV range.
    hsv_low = np.array([H_LOW, S_LOW, V_LOW])
    hsv_high = np.array([H_HIGH, S_HIGH, V_HIGH])
    mask = cv2.inRange(hsv_image, hsv_low, hsv_high)

    # Apply a region of interest (ROI) to the mask
    roi = get_roi((width, height))
    mask_roi = mask * roi
    show_image(mask_roi, 'Mask Image', cmap='gray')

    # Step 3: Perform thinning on the masked image
    thinned_image = thin(mask_roi).astype(np.uint8)
    show_image(thinned_image, 'Thinned Image', cmap='gray')

    # Step 4: Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(thinned_image, 0.5, np.pi / 180, 25, None, 60, 120)
    line_image = resize_image.copy()

    if lines is None:
        raise ValueError('No lines detected. '
                         'Please adjust proper arguments in cv2.HoughLinesP(...).')

    for line in lines:
        for begin_x, begin_y, end_x, end_y in line:
            line_image = cv2.line(resize_image, (begin_x, begin_y), (end_x, end_y), color=(255, 0, 0), thickness=2)

    # Step 5: Compute the vanishing point from the detected lines
    vanishing_point = get_vanishing_point(lines).flatten()
    print('Vanishing point:', vanishing_point)

    vanishing_point_line_image = cv2.circle(line_image, (int(vanishing_point[0]), int(vanishing_point[1])),
                                            10, (0, 0, 255), thickness=-1)
    show_image(vanishing_point_line_image, 'Vanishing Point + Lines')

    # Step 6: Define trapezoid for perspective transform around the vanishing point
    top = vanishing_point[1] + TOP_OFFSET
    bottom = height + BOTTOM_OFFSET

    # Calculate the trapezoid's corners around the vanishing point
    top_left = np.array([vanishing_point[0] - BOTTOM_WIDTH / 2, top])
    top_right = np.array([vanishing_point[0] + BOTTOM_WIDTH / 2, top])
    bottom_right = on_line(top_right, vanishing_point, bottom)
    bottom_left = on_line(top_left, vanishing_point, bottom)

    src_points = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.float32)
    dst_points = np.array([[0, height], [0, 0], [width, 0],  [width, height]], dtype=np.float32)
    print('Source points:', src_points)
    print('Destination points:', dst_points)

    trapezoid = cv2.polylines(rgb_image, [src_points.astype(np.int32)], True, (0, 0, 255), thickness=5)
    show_image(trapezoid, 'Trapezoid')

    # Step 7: Compute perspective transform and apply it for bird's-eye view (BEV)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(rgb_image, M, (width, height))
    print('M:', M)
    show_image(bev_image, 'BEV Image')

    # Step 8: Compute inverse perspective transform and verify unwarping
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    unwarped_image = cv2.warpPerspective(bev_image, M_inv, (width, height))
    print('M_inv:', M_inv)
    show_image(unwarped_image, 'Unwarped Image')


if __name__ == '__main__':
    main()
