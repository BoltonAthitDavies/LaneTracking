#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudastereo.hpp>

#include <yaml-cpp/yaml.h>
#include <memory>
#include <mutex>
#include <chrono>
#include <string>
#include <vector>

class StereoFisheye2Depth : public rclcpp::Node
{
public:
    StereoFisheye2Depth() : Node("stereo_fisheye2depth_node")
    {
        // Check CUDA availability
        cuda_available_ = cv::cuda::getCudaEnabledDeviceCount() > 0;
        scale_ = 0.5;

        // Initialize parameters
        declare_parameters();
        
        // Initialize variables
        init_variables();
        
        // Get calibration paths
        path_calibration();
        
        // Load existing calibration
        load_calibration();
        
        // Initialize video capture
        init_video_capture();
        
        // Initialize publishers
        init_publishers();
        
        // Initialize GPU resources if available
        if (cuda_available_) {
            RCLCPP_INFO(this->get_logger(), "✅ CUDA enabled! Found %d CUDA device(s)", 
                       cv::cuda::getCudaEnabledDeviceCount());
            init_gpu_resources();
        } else {
            RCLCPP_WARN(this->get_logger(), "⚠️  WARNING: CUDA not available! Falling back to CPU processing.");
            return;
        }
        
        // Create timer for processing
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // ~30 FPS
            std::bind(&StereoFisheye2Depth::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "Stereo Fisheye to Depth Node has been started");
    }
    
    ~StereoFisheye2Depth()
    {
        if (left_cap_.isOpened()) left_cap_.release();
        if (right_cap_.isOpened()) right_cap_.release();
        cv::destroyAllWindows();
    }

private:
    // Node variables
    bool cuda_available_;
    double scale_;
    std::mutex image_lock_;
    
    // Calibration results
    cv::Mat left_K_, left_D_, right_K_, right_D_;
    cv::Mat R_, T_, E_, F_;
    cv::Size img_shape_;
    
    // GPU resources
    cv::cuda::GpuMat gpu_map1_l_, gpu_map2_l_;
    cv::cuda::GpuMat gpu_map1_r_, gpu_map2_r_;
    cv::Ptr<cv::cuda::StereoBM> stereo_bm_;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> disp_bf_;
    
    // Video capture
    cv::VideoCapture left_cap_, right_cap_;
    
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr left_undist_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr right_undist_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr depth_compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Paths
    std::string path_yaml_;
    
    void declare_parameters()
    {
        this->declare_parameter("file_name_yaml", "matlab_calibration_resize.cpp.yaml");
        this->declare_parameter("show_images", false);
        this->declare_parameter("pub_unistortion_image", false);
        this->declare_parameter("compress_depth", true);
        this->declare_parameter("resize_image", true);
        this->declare_parameter("num_disparities", 256);
        this->declare_parameter("block_size", 15);
        this->declare_parameter("max_depth", 400.0);
        this->declare_parameter("min_depth", 0.0);
    }
    
    void init_variables()
    {
        left_K_ = cv::Mat::zeros(3, 3, CV_64F);
        left_D_ = cv::Mat::zeros(1, 4, CV_64F);
        right_K_ = cv::Mat::zeros(3, 3, CV_64F);
        right_D_ = cv::Mat::zeros(1, 4, CV_64F);
        R_ = cv::Mat::eye(3, 3, CV_64F);
        T_ = cv::Mat::zeros(3, 1, CV_64F);
    }
    
    void path_calibration()
    {
        std::string pkg_name = "robot_camera";
        std::string pkg_share_path = ament_index_cpp::get_package_share_directory(pkg_name);
        
        // Extract workspace path
        size_t install_pos = pkg_share_path.find("install");
        std::string ws_path = pkg_share_path.substr(0, install_pos);
        
        std::string file_name_yaml = this->get_parameter("file_name_yaml").as_string();
        path_yaml_ = ws_path + "src/" + pkg_name + "/config/" + file_name_yaml;
    }
    
    void load_calibration()
    {
        try {
            if (std::filesystem::exists(path_yaml_)) {
                YAML::Node calib_data = YAML::LoadFile(path_yaml_);
                
                // Load calibration matrices
                auto left_k_seq = calib_data["left_K"];
                auto left_d_seq = calib_data["left_D"];
                auto right_k_seq = calib_data["right_K"];
                auto right_d_seq = calib_data["right_D"];
                auto r_seq = calib_data["R"];
                auto t_seq = calib_data["T"];
                auto img_shape_seq = calib_data["image_shape"];
                
                // Convert YAML sequences to OpenCV matrices
                std::vector<double> left_k_vec = left_k_seq.as<std::vector<double>>();
                std::vector<double> left_d_vec = left_d_seq.as<std::vector<double>>();
                std::vector<double> right_k_vec = right_k_seq.as<std::vector<double>>();
                std::vector<double> right_d_vec = right_d_seq.as<std::vector<double>>();
                std::vector<double> r_vec = r_seq.as<std::vector<double>>();
                std::vector<double> t_vec = t_seq.as<std::vector<double>>();
                std::vector<int> img_shape_vec = img_shape_seq.as<std::vector<int>>();
                
                left_K_ = cv::Mat(3, 3, CV_64F, left_k_vec.data()).clone();
                left_D_ = cv::Mat(1, 4, CV_64F, left_d_vec.data()).clone();
                right_K_ = cv::Mat(3, 3, CV_64F, right_k_vec.data()).clone();
                right_D_ = cv::Mat(1, 4, CV_64F, right_d_vec.data()).clone();
                R_ = cv::Mat(3, 3, CV_64F, r_vec.data()).clone();
                T_ = cv::Mat(3, 1, CV_64F, t_vec.data()).clone();
                
                img_shape_ = cv::Size(img_shape_vec[0], img_shape_vec[1]);
                
                RCLCPP_INFO(this->get_logger(), "Loaded existing calibration and generated maps");
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration: %s", e.what());
        }
    }
    
    void init_video_capture()
    {
        std::string left_cam_pipeline = 
            "v4l2src device=/dev/video0 ! "
            "video/x-raw, width=1920, height=1080, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink";
            
        std::string right_cam_pipeline = 
            "v4l2src device=/dev/video1 ! "
            "video/x-raw, width=1920, height=1080, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink";
        
        left_cap_.open(left_cam_pipeline, cv::CAP_GSTREAMER);
        right_cap_.open(right_cam_pipeline, cv::CAP_GSTREAMER);
        
        if (!left_cap_.isOpened() || !right_cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open video devices!");
            return;
        }
    }
    
    void init_publishers()
    {
        bool pub_undistortion_image = this->get_parameter("pub_unistortion_image").as_bool();
        bool compress_depth = this->get_parameter("compress_depth").as_bool();
        
        if (pub_undistortion_image) {
            left_undist_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>
                ("/cam0_undis/image_raw/compressed", 1);
            right_undist_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>
                ("/cam1_undis/image_raw/compressed", 1);
        }
        
        if (compress_depth) {
            depth_compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>
                ("/depth/image_raw/compressed", 1);
        } else {
            depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>
                ("/depth/image_raw", 1);
        }
    }
    
    void init_gpu_resources()
    {
        // Create undistortion maps
        cv::Mat map1_l, map2_l, map1_r, map2_r;
        cv::fisheye::initUndistortRectifyMap(
            left_K_, left_D_, cv::Mat::eye(3, 3, CV_64F), left_K_, 
            img_shape_, CV_32FC1, map1_l, map2_l);
        cv::fisheye::initUndistortRectifyMap(
            right_K_, right_D_, cv::Mat::eye(3, 3, CV_64F), right_K_, 
            img_shape_, CV_32FC1, map1_r, map2_r);
        
        // Upload maps to GPU
        gpu_map1_l_.upload(map1_l);
        gpu_map2_l_.upload(map2_l);
        gpu_map1_r_.upload(map1_r);
        gpu_map2_r_.upload(map2_r);
        
        // Initialize stereo matchers
        init_gpu_stereo_matchers();
    }
    
    void init_gpu_stereo_matchers()
    {
        int num_disparities = this->get_parameter("num_disparities").as_int();
        int block_size = this->get_parameter("block_size").as_int();
        
        // BM matcher
        stereo_bm_ = cv::cuda::createStereoBM(num_disparities, block_size);
        
        // Disparity Bilateral Filter
        disp_bf_ = cv::cuda::createDisparityBilateralFilter(num_disparities, 1, 1);
    }
    
    void timer_callback()
    {
        cv::Mat left_frame, right_frame;
        bool ret_l = left_cap_.read(left_frame);
        bool ret_r = right_cap_.read(right_frame);
        
        if (!ret_l || !ret_r) {
            RCLCPP_WARN(this->get_logger(), "Failed to read frames from camera.");
            return;
        }
        
        // Get timestamp
        auto timestamp = this->now();
        
        // Process frames
        cv::Mat left_u, right_u, depth;
        process_frame(left_frame, right_frame, left_u, right_u, depth);
        
        // Publish results
        publish_compressed_images(left_u, right_u, depth, timestamp);
    }
    
    void process_frame(const cv::Mat& left_frame, const cv::Mat& right_frame,
                      cv::Mat& left_u, cv::Mat& right_u, cv::Mat& depth)
    {
        bool show_images = this->get_parameter("show_images").as_bool();
        
        // Undistort frames using GPU
        cv::cuda::GpuMat d_left_u, d_right_u;
        undistort_frames_gpu(left_frame, right_frame, d_left_u, d_right_u);
        
        if (show_images) {
            cv::Mat left_u_cpu, right_u_cpu;
            d_left_u.download(left_u_cpu);
            d_right_u.download(right_u_cpu);
            
            int new_w = static_cast<int>(left_u_cpu.cols * scale_);
            int new_h = static_cast<int>(left_u_cpu.rows * scale_);
            cv::resize(left_u_cpu, left_u_cpu, cv::Size(new_w, new_h));
            cv::resize(right_u_cpu, right_u_cpu, cv::Size(new_w, new_h));
            
            cv::imshow("Left Undistorted", left_u_cpu);
            cv::imshow("Right Undistorted", right_u_cpu);
        }
        
        // Compute disparity
        cv::Mat disp;
        compute_disparity_gpu(d_left_u, d_right_u, disp);
        
        if (show_images) {
            cv::Mat disp_vis;
            int new_w = static_cast<int>(disp.cols * scale_);
            int new_h = static_cast<int>(disp.rows * scale_);
            cv::resize(disp, disp_vis, cv::Size(new_w, new_h));
            cv::imshow("Disparity", disp_vis);
        }
        
        // Compute depth
        disparity_to_depth(disp, depth);
        
        if (show_images) {
            visualize_depth(depth);
            cv::waitKey(1);
        }
        
        // Download results from GPU
        d_left_u.download(left_u);
        d_right_u.download(right_u);
    }
    
    void undistort_frames_gpu(const cv::Mat& left_frame, const cv::Mat& right_frame,
                             cv::cuda::GpuMat& d_left_u, cv::cuda::GpuMat& d_right_u)
    {
        // Upload to GPU
        cv::cuda::GpuMat d_left, d_right;
        d_left.upload(left_frame);
        d_right.upload(right_frame);
        
        // Resize if needed
        bool resize_image = this->get_parameter("resize_image").as_bool();
        if (resize_image) {
            cv::cuda::resize(d_left, d_left, img_shape_);
            cv::cuda::resize(d_right, d_right, img_shape_);
        }
        
        // Undistort
        cv::cuda::remap(d_left, d_left_u, gpu_map1_l_, gpu_map2_l_, cv::INTER_LINEAR);
        cv::cuda::remap(d_right, d_right_u, gpu_map1_r_, gpu_map2_r_, cv::INTER_LINEAR);
    }
    
    void compute_disparity_gpu(const cv::cuda::GpuMat& d_left_u, const cv::cuda::GpuMat& d_right_u,
                              cv::Mat& disp)
    {
        // Convert to grayscale
        cv::cuda::GpuMat d_left_gray, d_right_gray;
        cv::cuda::cvtColor(d_left_u, d_left_gray, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(d_right_u, d_right_gray, cv::COLOR_BGR2GRAY);
        
        // Compute disparity
        cv::cuda::GpuMat d_disp;
        stereo_bm_->compute(d_left_gray, d_right_gray, d_disp);
        
        // Download result
        d_disp.download(disp);
    }
    
    void disparity_to_depth(const cv::Mat& disparity, cv::Mat& depth)
    {
        double fx = left_K_.at<double>(0, 0);
        double baseline = std::abs(T_.at<double>(0, 0));
        
        // Convert disparity to depth
        depth = cv::Mat::zeros(disparity.size(), CV_32F);
        for (int y = 0; y < disparity.rows; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                float disp_val = disparity.at<float>(y, x);
                if (disp_val > 0) {
                    depth.at<float>(y, x) = static_cast<float>((fx * baseline) / (disp_val + 1e-6));
                }
            }
        }
    }
    
    void visualize_depth(cv::Mat& depth)
    {
        int w = depth.cols;
        int h = depth.rows;
        
        // Define keypoints
        std::vector<std::pair<std::string, cv::Point>> keypoints = {
            {"Center", cv::Point(w / 2, h / 2)},
            {"Left", cv::Point(w / 4, h / 2)},
            {"Right", cv::Point(3 * w / 4, h / 2)},
            {"Top-Left", cv::Point(0, 0)},
            {"Bottom-Right", cv::Point(w - 1, h - 1)}
        };
        
        for (const auto& kp : keypoints) {
            const std::string& label = kp.first;
            const cv::Point& pt = kp.second;
            
            float depth_value = depth.at<float>(pt.y, pt.x);
            std::string text = label + ": " + std::to_string(depth_value).substr(0, 4) + " m";
            
            // Calculate text size
            int baseline;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
            
            // Draw background rectangle
            cv::Point top_left(pt.x + 5, pt.y + 5);
            cv::Point bottom_right(pt.x + 5 + text_size.width, pt.y + 5 + text_size.height + baseline);
            cv::rectangle(depth, top_left, bottom_right, cv::Scalar(0, 0, 0), -1);
            
            // Draw text
            cv::putText(depth, text, cv::Point(pt.x + 5, pt.y + 5 + text_size.height),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        
        // Show depth image
        cv::Mat depth_vis;
        int new_w = static_cast<int>(depth.cols * scale_);
        int new_h = static_cast<int>(depth.rows * scale_);
        cv::resize(depth, depth_vis, cv::Size(new_w, new_h));
        cv::imshow("Depth", depth_vis);
    }
    
    void publish_compressed_images(const cv::Mat& left_u, const cv::Mat& right_u, 
                                  const cv::Mat& depth, const rclcpp::Time& timestamp)
    {
        try {
            bool pub_undistortion_image = this->get_parameter("pub_unistortion_image").as_bool();
            bool compress_depth = this->get_parameter("compress_depth").as_bool();
            
            if (pub_undistortion_image) {
                // Publish undistorted images
                auto left_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", left_u).toCompressedImageMsg();
                auto right_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", right_u).toCompressedImageMsg();
                
                left_msg->header.stamp = timestamp;
                right_msg->header.stamp = timestamp;
                left_msg->header.frame_id = "cam0_undis";
                right_msg->header.frame_id = "cam1_undis";
                
                left_undist_pub_->publish(*left_msg);
                right_undist_pub_->publish(*right_msg);
            }
            
            // Check for valid depth data
            cv::Mat finite_mask;
            cv::compare(depth, depth, finite_mask, cv::CMP_EQ); // Check for finite values
            
            if (cv::countNonZero(finite_mask) > 0) {
                if (compress_depth) {
                    // Normalize depth for compressed publication
                    cv::Mat normalized_depth;
                    cv::normalize(depth, normalized_depth, 0, 255, cv::NORM_MINMAX, CV_8U);
                    
                    auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", normalized_depth).toCompressedImageMsg();
                    depth_msg->header.stamp = timestamp;
                    depth_msg->header.frame_id = "cam0_depth";
                    depth_compressed_pub_->publish(*depth_msg);
                } else {
                    // Publish raw depth
                    auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", depth).toImageMsg();
                    depth_msg->header.stamp = timestamp;
                    depth_msg->header.frame_id = "cam0_depth";
                    depth_pub_->publish(*depth_msg);
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to publish compressed images: %s", e.what());
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoFisheye2Depth>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}