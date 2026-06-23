import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package name (replace 'my_robot_package' with your actual package name)
    package_name = 'robot_description'
    
    # Path to URDF file
    urdf_file = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        'ackermann_robot.urdf'
    )
    
    # Path to RViz config file (optional)
    rviz_config_file = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'robot_rviz.rviz'
    )
    
    # Read URDF file
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()
    
    # Robot State Publisher Node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': False
        }]
    )
    
    # Joint State Publisher Node (for manual joint control in RViz)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )
    
    # Joint State Publisher GUI Node (optional - for GUI control)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )
    
    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else []
    )
    
    # Create and return launch description
    return LaunchDescription([
        robot_state_publisher_node,
        # joint_state_publisher_node,
        # joint_state_publisher_gui_node,  # Uncomment if you want GUI control
        rviz_node
    ])