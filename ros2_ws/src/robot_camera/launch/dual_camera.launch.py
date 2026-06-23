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
            'video_device': '/dev/video0',
            # 'camera_info_url': path_yaml_cam0,
            'pixel_format': 'UYVY',
            'image_size': [1920, 1080],
            'frame_id': 'camera0_frame',
            'output_encoding': 'yuv422',
        }]
    )

    cam1_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace='cam1',
        parameters=[{
            'video_device': '/dev/video1',
            'pixel_format': 'UYVY',
            'image_size': [1920, 1080],
            'frame_id': 'camera0_frame',
            'output_encoding': 'yuv422',
        }]
    )

    launch_description = LaunchDescription()
    launch_description.add_action(cam0_node)
    launch_description.add_action(cam1_node)

    return launch_description
