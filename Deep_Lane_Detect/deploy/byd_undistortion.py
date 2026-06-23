import os
import yaml
import numpy as np
import cv2

class Undistortion():
    def __init__(self):
        self._init_cv()
        self._load_calibration()
        self._init_gpu_if_available()
        self._init_camera()
        # Removed internal FPS trackers as we will do it in the main loop
        
    def _init_cv(self):
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.scale = 0.5
        self.left_K = self.left_D = self.right_K = self.right_D = None
        self.R = self.T = self.E = self.F = None
        self.img_shape = None
        self.path_yaml = None
        self.frame = None

    def _load_calibration(self):
        try:
            # IMPORTANT: Ensure this path is correct for your system
            self.path_yaml = "/home/nvidia/LaneTracking/ros2_ws/src/robot_camera/config/matlab_calibration.yaml"
            if not os.path.exists(self.path_yaml):
                raise FileNotFoundError(f"{self.path_yaml} not found")
            with open(self.path_yaml, 'r') as f:
                calib_data = yaml.safe_load(f)
            self.left_K = np.array(calib_data['left_K'])
            self.left_D = np.array(calib_data['left_D'])
            self.right_K = np.array(calib_data['right_K'])
            self.right_D = np.array(calib_data['right_D'])
            self.R = np.array(calib_data['R'])
            self.T = np.array(calib_data['T'])
            self.img_shape = tuple(calib_data['image_shape'])
            print("Loaded calibration successfully.")
        except Exception as e:
            print(f"Failed to load calibration: {e}")

    def _init_gpu_if_available(self):
        if self.cuda_available:
            print(f"✅ CUDA enabled for OpenCV! Found {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            self._init_gpu_resources()
        else:
            print("⚠️ CUDA not available for OpenCV! Using CPU fallback.")

    def _init_gpu_resources(self):
        # Initialize GPU undistortion maps
        map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(self.left_K, self.left_D, np.eye(3), self.left_K, self.img_shape, cv2.CV_32FC1)
        # map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(self.right_K, self.right_D, np.eye(3), self.right_K, self.img_shape, cv2.CV_32FC1)

        self.gpu_map1_l, self.gpu_map2_l = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        # self.gpu_map1_r, self.gpu_map2_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        self.gpu_map1_l.upload(map1_l)
        self.gpu_map2_l.upload(map2_l)
        # self.gpu_map1_r.upload(map1_r)
        # self.gpu_map2_r.upload(map2_r)

    def _init_camera(self):
        pipeline = " ! ".join(["v4l2src device=/dev/video0",
                            "video/x-raw, width=1920, height=1080, framerate=30/1",
                            # "nvvidconv",  # <--- FAST (GPU/HARDWARE)
                            "videoconvert", # <--- ADD THIS LINE
                            "video/x-raw, format=(string)BGR",
                            "appsink"
                            ])
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def undistort_frame(self, left_frame):
        """
        Undistorts a single frame using pre-computed GPU maps.
        Returns only the display-sized (640x360) undistorted image.
        """
        d_left = cv2.cuda_GpuMat()
        d_left.upload(left_frame)

        # Perform GPU-accelerated remapping
        d_left_u = cv2.cuda.remap(d_left, self.gpu_map1_l, self.gpu_map2_l, cv2.INTER_LINEAR)

        # Resize for display
        d_left_u = cv2.cuda.resize(d_left_u, (480, 320))
        
        # Download the final image from GPU to CPU
        return d_left_u