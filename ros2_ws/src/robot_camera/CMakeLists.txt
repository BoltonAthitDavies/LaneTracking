cmake_minimum_required(VERSION 3.5)
project(robot_camera)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################ FIND DEPENDENCIES  #################
find_package(ament_cmake REQUIRED)		# cpp (optional)
find_package(rclcpp REQUIRED)			# cpp (optional)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(ament_cmake_python REQUIRED)   	# python (optional)
find_package(rclpy REQUIRED)                	# python (optional)
######################################################

# Check for CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  message(STATUS "CUDA found: ${CUDA_VERSION}")
  add_definitions(-DWITH_CUDA)
else()
  message(WARNING "CUDA not found - GPU acceleration will not be available")
endif()

# Find yaml-cpp
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Install Python Module
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/dummy_script.py
  scripts/stereo_fisheye2depth.py
  scripts/stereo_fisheye2depth_pipeline.py
  scripts/stereo_fisheye2depth_pipeline_original.py
  scripts/stereo_fisheye2depth_lane_detect_pipeline.py
  scripts/stereo_fisheye2depth_pipeline_add_sgbm_wls.py
  scripts/mono_fisheye_undistort.py
  scripts/left_cam_recorder_node.py
  scripts/stereo_cam_recorder_node.py
  scripts/config.py
  scripts/trt_infer.py
  scripts/lane_detect.py
  scripts/lane_detect_image.py
  scripts/lane_detect_no_depth.py
  scripts/camera_watchdog_node.py
  scripts/byd_trt_infer.py
  scripts/byd_lane_detect_image.py
  scripts/udp_sender_node.py
  DESTINATION lib/${PROJECT_NAME}
)


# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(cpp_node_test src/cpp_node.cpp)
ament_target_dependencies(cpp_node_test rclcpp)
add_executable(stereo2depth_node src/stereo2depth_node.cpp)
ament_target_dependencies(stereo2depth_node 
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  ament_index_cpp
)
add_executable(stereo_undistort_node src/stereo_undistort_node.cpp)
ament_target_dependencies(stereo_undistort_node
  rclcpp
  sensor_msgs
  cv_bridge
  ament_index_cpp
)

add_executable(stereo_fisheye2depth_pipeline src/stereo_fisheye2depth_pipeline.cpp)
ament_target_dependencies(stereo_fisheye2depth_pipeline
  rclcpp
  sensor_msgs
  cv_bridge
  ament_index_cpp
)

add_executable(cpp_mono_fisheye_undistort src/cpp_mono_fisheye_undistort.cpp)
ament_target_dependencies(cpp_mono_fisheye_undistort
  rclcpp
  sensor_msgs
  cv_bridge
  ament_index_cpp
)

# Target include directories
target_include_directories(stereo_fisheye2depth_pipeline PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(cpp_mono_fisheye_undistort PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

# Target compile features
target_compile_features(stereo_fisheye2depth_pipeline PUBLIC c_std_99 cxx_std_17)
target_compile_features(cpp_mono_fisheye_undistort PUBLIC c_std_99 cxx_std_17)

# Link OpenCV libraries
target_link_libraries(stereo2depth_node yaml-cpp)
target_link_libraries(stereo_undistort_node
  ${OpenCV_LIBRARIES}
  yaml-cpp
)
target_link_libraries(stereo_fisheye2depth_pipeline
  ${OpenCV_LIBRARIES}
  yaml-cpp
)
target_link_libraries(cpp_mono_fisheye_undistort
  ${OpenCV_LIBRARIES}
  yaml-cpp
)
# Install Cpp executables
install(TARGETS
  cpp_node_test
  stereo2depth_node
  stereo_undistort_node
  stereo_fisheye2depth_pipeline
  cpp_mono_fisheye_undistort
  DESTINATION lib/${PROJECT_NAME}
)

################ INSTALL LAUNCH, ETC #################
install(DIRECTORY
  # add directories here
  config
  launch
  model
  DESTINATION share/${PROJECT_NAME})

ament_package()
