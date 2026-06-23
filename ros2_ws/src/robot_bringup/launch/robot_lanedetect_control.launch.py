from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
# import xacro    
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# for display robot_state_publisher and fix something
    
def generate_launch_description():
    
    pkg_robot_description = get_package_share_directory('robot_description')

    # Declare launch argument for steering mode
    log_mode_arg = DeclareLaunchArgument(
        'log_feedback',
        default_value='false',
        description='Enable log feedback velocity and steering from robot by canbus'
    )
    interface_type_arg = DeclareLaunchArgument(
        'interface_type',
        default_value='canable',
        description='Select interface type of canbus'
    )
    show_images_arg = DeclareLaunchArgument(
        'show_images',
        default_value='false',
        description='Visulize camera and lane detection'
    )
    resize_image_arg = DeclareLaunchArgument(
        'resize_image',
        default_value='true',
        description='Resize image after unistortion'
    )
    ori_width_arg = DeclareLaunchArgument(
        'ori_width',
        default_value='480',
        description='Width of image'
    )
    ori_height_arg = DeclareLaunchArgument(
        'ori_height',
        default_value='320',
        description='Height of image'
    )
    engine_path_arg = DeclareLaunchArgument(
        'engine_path',
        default_value='model/tusimple_res34_fix_num_col_36.engine',
        # default_value='model/tusimple_res34_bend_25_v1_480x320.engine',
        description='Path of engine file'
    )
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='config/tusimple_res34_fix_num_col_36.py',
        # default_value='config/tusimple_res34_bend_25_v1_480x320.py',
        description='Path of config file'
    )
    target_speed_arg = DeclareLaunchArgument(
        'target_speed',
        default_value='0.864',
        description='Speed of robot'
    )
    kp_steer_arg = DeclareLaunchArgument(
        'kp_steer',
        default_value='500.0',
        description='Kp of offset between center image and center lane'
    )
    ki_steer_arg = DeclareLaunchArgument(
        'ki_steer',
        default_value='0.0',
        description='Kp of offset between center image and center lane'
    )
    kd_steer_arg = DeclareLaunchArgument(
        'kd_steer',
        default_value='0.0',
        description='Kp of offset between center image and center lane'
    )
    cam0_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace='cam0',
        parameters=[{
            'video_device': '/dev/video0',
            'pixel_format': 'UYVY',
            'image_size': [1920, 1080],
            'frame_id': 'camera0_frame',
            'output_encoding': 'yuv422',
        }]
    )

    robot_can_interface_node = Node(
        package="robot_can_interface",
        executable="robot_can_interface_node.py",
        name="robot_can_interface_node",
        parameters=[{"log_feedback": LaunchConfiguration("log_feedback")},
                    {"interface_type": LaunchConfiguration("interface_type")}]
    )

    undistortion_node = Node(
        package="robot_camera",
        executable="mono_fisheye_undistort.py",
        name="undistortion_node",
        parameters=[{"show_images": LaunchConfiguration("show_images")},
                    {"resize_image": LaunchConfiguration("resize_image")},
                    {"ori_width": LaunchConfiguration("ori_width")},
                    {"ori_height": LaunchConfiguration("ori_height")}]
    )

    # lane_detect_node = Node(
    #     package="robot_camera",
    #     executable="lane_detect_image.py",
    #     name="lane_detect_node",
    #     parameters=[{"show_images": LaunchConfiguration("show_images")},
    #                 {"ori_width": LaunchConfiguration("ori_width")},
    #                 {"ori_height": LaunchConfiguration("ori_height")},
    #                 {"engine_path": LaunchConfiguration("engine_path")},
    #                 {"config_path": LaunchConfiguration("config_path")}]
    # )

    lane_detect_node = Node(
        package="robot_camera",
        executable="byd_lane_detect_image.py",
        name="lane_detect_node",
        parameters=[{"show_images": LaunchConfiguration("show_images")},
                    {"engine_path": LaunchConfiguration("engine_path")},
                    {"config_path": LaunchConfiguration("config_path")}]
    )

    pid_controller_node = Node(
        package="robot_controller",
        executable="pid_controller.py",
        name="pid_controller_node",
        parameters=[{"log_feedback": LaunchConfiguration("log_feedback")},
                    {"target_speed": LaunchConfiguration("target_speed")},
                    {"kp_steer": LaunchConfiguration("kp_steer")},
                    {"ki_steer": LaunchConfiguration("ki_steer")},
                    {"kd_steer": LaunchConfiguration("kd_steer")}]
    )

    lowpass_filter_node = Node(
        package="robot_controller",
        executable="lowpass_filter.py",
        name="lowpass_filter_node"
    )

    launch_description = LaunchDescription()
    launch_description.add_action(log_mode_arg)
    launch_description.add_action(interface_type_arg)
    launch_description.add_action(show_images_arg)
    launch_description.add_action(resize_image_arg)
    launch_description.add_action(ori_width_arg)
    launch_description.add_action(ori_height_arg)
    launch_description.add_action(engine_path_arg)
    launch_description.add_action(config_path_arg)
    launch_description.add_action(target_speed_arg)
    launch_description.add_action(kp_steer_arg)
    launch_description.add_action(ki_steer_arg)
    launch_description.add_action(kd_steer_arg)
    
    launch_description.add_action(lane_detect_node)
    
    launch_description.add_action(robot_can_interface_node)
    launch_description.add_action(lowpass_filter_node)
    launch_description.add_action(pid_controller_node)

    return launch_description