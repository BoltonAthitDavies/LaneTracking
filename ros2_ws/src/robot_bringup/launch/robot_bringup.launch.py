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

    # launch rviz
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    pkg_robot_description,
                    "launch",
                    "rviz.launch.py"
                )
            ]
        )
    )

    launch_teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        namespace='',
        output='screen',
        prefix='xterm -e')

    robot_can_interface_node = Node(
        package="robot_can_interface",
        executable="robot_can_interface_node.py",
        name="robot_can_interface_node",
        parameters=[{"log_feedback": LaunchConfiguration("log_feedback")},
                    {"interface_type": LaunchConfiguration("interface_type")}]
    )

    # # 🔑 Run your setting script once (reset yaw)
    # hwt101ct_reset_once = ExecuteProcess(
    #     cmd=["python3", "/home/ubuntu/LaneTracking/ros2_ws/src/hwt101ct_tilt_angle_sensor/hwt101ct_setting.py", "reset"],
    #     output="screen"
    # )

    hwt101ct_yaw_publisher = Node(
        package='hwt101ct_tilt_angle_sensor',
        executable='hwt101ct_yaw_publisher.py',
        name="hwt101ct_yaw_node",
    )

    # # 👇 hwt101ct_yaw_publisher will only start after hwt101ct_reset_once finishes
    # hwt101ct_yaw_publisher = RegisterEventHandler(
    #     OnProcessExit(
    #         target_action=hwt101ct_reset_once,
    #         on_exit=[
    #             Node(
    #                 package="hwt101ct_tilt_angle_sensor",
    #                 executable="hwt101ct_yaw_publisher.py",
    #                 name="hwt101ct_yaw_node"
    #             )
    #         ]
    #     )
    # )

    steering_model_node = Node(
        package="robot_controller",
        executable="steering_model_node.py",
        name="steering_model_node",
    )

    odometry_node = Node(
        package="robot_odometry",
        executable="odometry_node.py",
        name="odometry_node"
    )

    camera_node = Node(
        package="robot_camera",
        executable="stereo_fisheye2depth_lane_detect_pipeline.py",
        name="camera_node"
    )

    # controller_server = Node(
    #     package="limo_controller",
    #     executable="controller_server.py",
    #     name="controller_server",
    #     parameters=[{"control_mode": LaunchConfiguration("control_mode")}]
    # )

    launch_description = LaunchDescription()
    launch_description.add_action(log_mode_arg)
    launch_description.add_action(interface_type_arg)
    launch_description.add_action(rviz)
    launch_description.add_action(launch_teleop)
    launch_description.add_action(robot_can_interface_node)
    # launch_description.add_action(hwt101ct_reset_once)    # run first
    launch_description.add_action(hwt101ct_yaw_publisher) # wait until reset finishes
    launch_description.add_action(steering_model_node)
    launch_description.add_action(odometry_node)
    # launch_description.add_action(camera_node)
    
    return launch_description