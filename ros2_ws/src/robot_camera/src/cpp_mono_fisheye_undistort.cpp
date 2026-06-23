#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/cudawarping.hpp> // For cv::cuda::remap
#include <opencv2/cudaarithm.hpp> // For cv::cuda::resize
#include <opencv2/core/cuda.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <tuple>
#include <mutex>

using namespace std::chrono_literals;

// Define the name of the package containing the config file
#define PACKAGE_NAME "robot_camera"

class StereoFisheye2Depth : public rclcpp::Node
{
public:
    StereoFisheye2Depth()
    : Node("stereo_fisheye2depth_node")
    {
        // === Initialization Methods ===
        declare_params();
        init_cv();
        load_calibration();
        init_ros_io();
        init_camera_streams();
        init_gpu_if_available();
        init_timer();

        // === FPS counter ===
        last_time_ = this->now();
        frame_count_ = 0;
        fps_ = 0.0;

        RCLCPP_INFO(this->get_logger(), "Stereo Fisheye to Depth Node initialized.");
    }

private:
    // === ROS 2 Members ===
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_undist_pub_img_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr left_undist_pub_comp_;
    rclcpp::TimerBase::SharedPtr timer_;

    // === OpenCV/CV-Bridge Members ===
    cv_bridge::CvImagePtr cv_ptr_;
    cv::VideoCapture left_cap_;
    std::mutex image_lock_;
    bool cuda_available_ = false;
    double scale_ = 0.5;

    // Calibration and Rectification
    cv::Mat left_K_, left_D_, right_K_, right_D_, R_, T_;
    cv::Size img_shape_;
    cv::Mat map1_l_, map2_l_; // CPU maps

    // GPU Buffers and maps
    cv::cuda::GpuMat gpu_map1_l_, gpu_map2_l_;

    // === Parameters ===
    std::string file_name_yaml_;
    bool show_images_;
    bool resize_image_;
    bool compress_undistort_;

    // === FPS Counter Members ===
    rclcpp::Time last_time_;
    int frame_count_;
    double fps_;

    // --- Initialization Methods ---
    void declare_params()
    {
        this->declare_parameter<std::string>("file_name_yaml", "matlab_calibration_resize.cpp.yaml");
        this->declare_parameter<bool>("show_images", false);
        this->declare_parameter<bool>("resize_image", true);
        this->declare_parameter<bool>("compress_undistort", true);

        // Get parameter values
        file_name_yaml_ = this->get_parameter("file_name_yaml").as_string();
        show_images_ = this->get_parameter("show_images").as_bool();
        resize_image_ = this->get_parameter("resize_image").as_bool();
        compress_undistort_ = this->get_parameter("compress_undistort").as_bool();
    }

    void init_cv()
    {
        // CV-Bridge pointer initialized in the main function (cv_ptr_)
    }

    void load_calibration()
    {
        try {
            std::string file_name = resize_image_ ? file_name_yaml_ : "matlab_calibration.cpp.yaml";
            std::string pkg_path = ament_index_cpp::get_package_share_directory(PACKAGE_NAME);
            std::string root_path = pkg_path; // Simpler pathing in C++
            std::string path_yaml = root_path + "/config/" + file_name;

            // Check if the file exists
            if (!std::ifstream(path_yaml).good()) {
                throw std::runtime_error("Calibration file not found at: " + path_yaml);
            }

            YAML::Node calib_data = YAML::LoadFile(path_yaml);

            // Helper to convert YAML sequence to cv::Mat (assuming 1D or 2D list)
            auto yaml_to_mat = [](const YAML::Node& node, int rows, int cols) -> cv::Mat {
                cv::Mat mat(rows, cols, CV_64F); // Using CV_64F (double) for calibration data
                int k = 0;
                for (const auto& val : node) {
                    mat.at<double>(k / cols, k % cols) = val.as<double>();
                    k++;
                }
                return mat;
            };

            // Load data
            std::vector<int> shape_vec = calib_data["image_shape"].as<std::vector<int>>();
            img_shape_ = cv::Size(shape_vec[0], shape_vec[1]); // (width, height)

            left_K_ = yaml_to_mat(calib_data["left_K"], 3, 3);
            left_D_ = yaml_to_mat(calib_data["left_D"], 1, 4); // D is 1x4 for fisheye
            right_K_ = yaml_to_mat(calib_data["right_K"], 3, 3);
            right_D_ = yaml_to_mat(calib_data["right_D"], 1, 4);
            R_ = yaml_to_mat(calib_data["R"], 3, 3);
            T_ = yaml_to_mat(calib_data["T"], 3, 1);

            RCLCPP_INFO(this->get_logger(), "Loaded calibration successfully from: %s", path_yaml.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration: %s", e.what());
        }
    }

    void init_ros_io()
    {
        rclcpp::QoS qos_profile(1);
        qos_profile.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        qos_profile.history(rclcpp::HistoryPolicy::KeepLast);

        if (compress_undistort_) {
            left_undist_pub_comp_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("/cam0_undis/image_raw", qos_profile);
        } else {
            left_undist_pub_img_ = this->create_publisher<sensor_msgs::msg::Image>("/cam0_undis/image_raw", qos_profile);
        }
    }

    void init_camera_streams()
    {
        // Use the GStreamer pipeline helper
        std::string pipeline_l = gstreamer_pipeline("/dev/video0");
        left_cap_.open(pipeline_l, cv::CAP_GSTREAMER);

        if (!left_cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "❌ Failed to open video devices.");
        }
    }

    void init_timer()
    {
        // 30 Hz timer
        timer_ = this->create_wall_timer(1ms * 33, std::bind(&StereoFisheye2Depth::timer_callback, this));
    }

    void init_gpu_if_available()
    {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cuda_available_ = true;
            RCLCPP_INFO(this->get_logger(), "✅ CUDA enabled! Found %d device(s)", cv::cuda::getCudaEnabledDeviceCount());
            init_gpu_resources();
        } else {
            RCLCPP_WARN(this->get_logger(), "⚠️ CUDA not available! Using CPU fallback (Note: CPU undistort is not fully implemented in the C++ version as the Python code favored GPU).");
            // In a full C++ conversion, you'd add CPU logic here. Sticking to the Python code's GPU preference.
        }
    }

    std::string gstreamer_pipeline(const std::string& device)
    {
        // Same pipeline logic as Python
        return "v4l2src device=" + device + " ! video/x-raw, width=1920, height=1080, framerate=30/1 ! "
               "videoconvert ! video/x-raw, format=BGR ! appsink";
    }

    // --- Timer Callback ---
    void timer_callback()
    {
        cv::Mat left_frame;
        if (!left_cap_.read(left_frame) || left_frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "⚠️ Frame read failed.");
            return;
        }

        // Get the timestamp before processing
        rclcpp::Time timestamp = this->now();

        // Process the frame (undistort)
        cv::Mat left_u = process_frame(left_frame);

        // Publish the result
        publish_images(left_u, timestamp);

        // === FPS counter ===
        frame_count_++;
        rclcpp::Time current_time = this->now();
        if ((current_time - last_time_).seconds() >= 1.0) { // update every 1 second
            fps_ = frame_count_ / (current_time - last_time_).seconds();
            RCLCPP_INFO(this->get_logger(), "📸 Processing FPS: %.2f", fps_);
            frame_count_ = 0;
            last_time_ = current_time;
        }
    }

    // --- GPU Processing ---
    void init_gpu_resources()
    {
        // Initialize the undistortion maps for the left camera (assuming the right camera maps were irrelevant to the mono_fisheye_undistort.py file's actual execution, which only processes left).
        cv::fisheye::initUndistortRectifyMap(
            left_K_, left_D_, cv::Mat::eye(3, 3, CV_64F), left_K_, img_shape_, CV_32FC1, map1_l_, map2_l_
        );

        // Upload maps to GPU
        gpu_map1_l_.upload(map1_l_);
        gpu_map2_l_.upload(map2_l_);
    }

    // --- Frame Processing ---
    cv::Mat process_frame(const cv::Mat& left_frame)
    {
        cv::Mat left_u_cpu;

        if (cuda_available_) {
            // Undistort using GPU
            cv::cuda::GpuMat left_u_gpu = undistort_frames_gpu(left_frame);
            // Download from GPU
            left_u_gpu.download(left_u_cpu);
        } else {
            // Fallback to CPU undistortion (needed for a complete conversion, though Python favored GPU)
            // Note: The Python code only had GPU undistortion logic called inside process_frame.
            // For a full conversion, a CPU path would go here.
            RCLCPP_ERROR(this->get_logger(), "No CUDA/GPU. Cannot process frame without CPU fallback logic.");
            return left_frame.clone(); // Return original image as placeholder
        }

        if (show_images_) {
            cv::Size new_size(left_frame.cols * scale_, left_frame.rows * scale_);
            display_results(left_u_cpu, new_size);
        }

        return left_u_cpu;
    }

    cv::cuda::GpuMat undistort_frames_gpu(const cv::Mat& left_frame)
    {
        cv::cuda::GpuMat d_left;
        d_left.upload(left_frame);

        if (resize_image_) {
            cv::cuda::resize(d_left, d_left, img_shape_);
        }

        cv::cuda::GpuMat d_left_u;
        // Use the GPU maps for remap
        cv::cuda::remap(d_left, d_left_u, gpu_map1_l_, gpu_map2_l_, cv::INTER_LINEAR);

        return d_left_u;
    }

    void display_results(const cv::Mat& left_u, const cv::Size& size)
    {
        // The Python code commented out resizing before showing, so we'll show the full undistorted image.
        cv::imshow("Left Undistorted", left_u);
        cv::waitKey(1);
    }

    // --- ROS Publishing ---
    void publish_images(const cv::Mat& left_u, const rclcpp::Time& timestamp)
    {
        try {
            // Create a CvImage to hold the data for cv_bridge
            cv_bridge::CvImage cv_img;
            cv_img.header.stamp = timestamp;
            cv_img.header.frame_id = "cam0_undis";
            cv_img.encoding = "bgr8"; // Assuming BGR format from gstreamer_pipeline
            cv_img.image = left_u;

            if (compress_undistort_) {
                // Publish CompressedImage
                if (left_undist_pub_comp_) {
                    auto comp_msg = cv_img.toCompressedImageMsg();
                    left_undist_pub_comp_->publish(*comp_msg);
                }
            } else {
                // Publish Image
                if (left_undist_pub_img_) {
                    auto img_msg = cv_img.toImageMsg();
                    left_undist_pub_img_->publish(*img_msg);
                }
            }
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to publish images: %s", e.what());
        }
    }
};

// --- Main Function ---
int main(int argc, char * argv[])
{
    // Initialize the ROS 2 system
    rclcpp::init(argc, argv);

    // Create the node and spin it (execute the timer callbacks)
    rclcpp::spin(std::make_shared<StereoFisheye2Depth>());

    // Shutdown the ROS 2 system
    rclcpp::shutdown();
    return 0;
}