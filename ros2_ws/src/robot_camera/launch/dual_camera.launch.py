import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # pkg_rover_camera = get_package_share_directory('robot_camera')
    # ws_path, _ = pkg_rover_camera.split('install')
    # file_name_yaml = 'cam0_calibration.yaml'
    # path_yaml_cam0 = os.path.join(ws_path, 'src', pkg_rover_camera, 'config', file_name_yaml)

    cam0_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace='cam0',
        parameters=[{
            'video_device': '/dev/video8',
            # 'camera_info_url': path_yaml_cam0,
            'pixel_format': 'UYVY',
            'image_size': [1920, 1080],
            'frame_id': 'camera0_frame',
            # 'output_encoding': 'yuv422'
            'output_encoding': 'rgb8',
        }]
    )

    cam1_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace='cam1',
        parameters=[{
            'video_device': '/dev/video9',
            'pixel_format': 'UYVY',
            'image_size': [1920, 1080],
            'frame_id': 'camera1_frame',
            # 'output_encoding': 'yuv422'
            'output_encoding': 'rgb8',
        }]
    )

    undistort_node = Node(
        package='robot_camera',
        executable='fisheye_undistort_node.py',
        name='fisheye_undistort_node',
    )

    imu_node = Node(
        package='robot_camera',
        executable='realsense_imu_publisher.py',
        name='realsense_imu_node',
        parameters=[{
            'gyro_rate_hz': 200.0,  # D455 motion module offers 200 or 400 Hz; matches the HWT101's 200 Hz
        }],
        remappings=[
            ('realsense_imu_publisher', 'hwt101ct_yaw_publisher'),
        ]
    )

    launch_description = LaunchDescription()
    launch_description.add_action(cam0_node)
    launch_description.add_action(cam1_node)
    launch_description.add_action(undistort_node)
    launch_description.add_action(imu_node)

    return launch_description
