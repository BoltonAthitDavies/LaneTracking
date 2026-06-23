"""
Author: Sippawit Thammawiset
Date: September 28, 2024.
File: get_pitch_yaw_from_vanishing.py
"""

import numpy as np

# The vanishing point got from "find_ipm.py"
vanishing_point = np.float32([226.15584, 79.75166, 1])

# The intrinsic matrix got from camera calibration
K = np.array([
    [204.012775, 0.000000, 211.861709],
    [0.000000, 204.594692, 123.687077],
    [0.000000, 0.000000, 1.000000],
])

K_inv = np.linalg.inv(K)
r3 = K_inv @ vanishing_point
r3 /= np.linalg.norm(r3)

pitch = np.arcsin(r3[1])
yaw = -np.arctan2(r3[0], r3[2])

print('> Pitch [deg]:', np.rad2deg(pitch))
print('> Yaw [deg]:', np.rad2deg(yaw))
