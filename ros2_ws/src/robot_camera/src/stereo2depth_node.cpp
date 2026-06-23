#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <mutex>
#include <chrono>
#include <string>
#include <filesystem>

class Stereo2Depth : public rclcpp::Node
{
public:
    Stereo2Depth() : Node("stereo2depth_node")
    {
        // Initialize member variables
        current_left_ = nullptr;
        current_right_ = nullptr;
        left_timestamp_ = this->now();
        right_timestamp_ = this->now();
        
        // Calibration matrices
        left_K_ = cv::Mat::zeros(3, 3, CV_64F);
        left_D_ = cv::Mat::zeros(4, 1, CV_64F);
        right_K_ = cv::Mat::zeros(3, 3, CV_64F);
        right_D_ = cv::Mat::zeros(4, 1, CV_64F);
        R_ = cv::Mat::zeros(3, 3, CV_64F);
        T_ = cv::Mat::zeros(3, 1, CV_64F);
        
        // Performance tracking
        last_process_time_ = std::chrono::high_resolution_clock::now();
        process_count_ = 0;
        fps_counter_ = 0;
        fps_start_time_ = std::chrono::high_resolution_clock::now();
        
        // Display configuration
        display_scale_ = 0.5;
        frame_skip_counter_ = 0;
        
        // Declare parameters
        this->declare_parameter("file_name_yaml", "matlab_calibration.cpp.yaml");
        this->declare_parameter("pub_undistort_images", false);
        this->declare_parameter("pub_disparity_map", false);
        this->declare_parameter("pub_depth_map", false);
        this->declare_parameter("show_images", false);
        this->declare_parameter("skip_frames", 1);
        
        // Stereo matching parameters
        this->declare_parameter("min_disparities", 0);
        this->declare_parameter("max_disparities", 16*12);
        this->declare_parameter("block_size", 5);
        this->declare_parameter("uniqueness_ratio", 5);
        this->declare_parameter("speckle_window_size", 50);
        this->declare_parameter("speckle_range", 16);
        this->declare_parameter("pre_filter_cap", 63);
        
        // WLS Filter parameters
        this->declare_parameter("use_wls_filter", true);
        this->declare_parameter("wls_lambda", 20000.0);
        this->declare_parameter("wls_sigma", 1.7);
        
        // Depth Estimation parameters
        this->declare_parameter("max_depth", 400.0);
        this->declare_parameter("min_depth", 0.0);
        
        // Get paths for calibration files
        path_calibration();
        
        // Setup QoS profile for real-time performance
        auto qos_profile = rclcpp::QoS(1)
            .reliability(rclcpp::ReliabilityPolicy::BestEffort)
            .history(rclcpp::HistoryPolicy::KeepLast);
        
        // Subscribers
        left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/cam0/image_raw", qos_profile,
            std::bind(&Stereo2Depth::left_callback, this, std::placeholders::_1));
        
        right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/cam1/image_raw", qos_profile,
            std::bind(&Stereo2Depth::right_callback, this, std::placeholders::_1));
        
        // Publishers
        left_undist_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/cam0_undis/image_raw", qos_profile);
        right_undist_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/cam1_undis/image_raw", qos_profile);
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/depth/image_raw", qos_profile);
        disparity_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/disparity/image_raw", qos_profile);
        
        // Timer for processing
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // ~30 Hz
            std::bind(&Stereo2Depth::process_images, this));
        
        // Load existing calibration if available
        load_calibration();
        
        // Initialize stereo matcher and WLS filter
        setup_stereo_matcher();
        
        RCLCPP_INFO(this->get_logger(), "Optimized Stereo2Depth Node with WLS Filter Initialized");
    }
    
    ~Stereo2Depth()
    {
        cv::destroyAllWindows();
    }

private:
    // Member variables
    std::shared_ptr<cv::Mat> current_left_;
    std::shared_ptr<cv::Mat> current_right_;
    rclcpp::Time left_timestamp_;
    rclcpp::Time right_timestamp_;
    std::mutex image_lock_;
    
    // Calibration results
    cv::Mat left_K_, left_D_, right_K_, right_D_;
    cv::Mat R_, T_, E_, F_;
    
    // Undistortion maps
    cv::Mat left_map1_, left_map2_, right_map1_, right_map2_;
    
    // Pre-allocated image buffers
    cv::Mat left_undist_buffer_, right_undist_buffer_;
    
    // Stereo matcher and WLS filter
    cv::Ptr<cv::StereoSGBM> stereo_matcher_left_;
    cv::Ptr<cv::StereoMatcher> stereo_matcher_right_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    double baseline_;
    std::string path_yaml_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point last_process_time_;
    int process_count_;
    int fps_counter_;
    std::chrono::high_resolution_clock::time_point fps_start_time_;
    
    // Display configuration
    double display_scale_;
    int frame_skip_counter_;
    
    // ROS2 components
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_undist_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_undist_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr disparity_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    void left_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            std::lock_guard<std::mutex> lock(image_lock_);
            current_left_ = std::make_shared<cv::Mat>(
                cv_bridge::toCvShare(msg, "bgr8")->image);
            left_timestamp_ = msg->header.stamp;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error converting left image: %s", e.what());
        }
    }
    
    void right_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            std::lock_guard<std::mutex> lock(image_lock_);
            current_right_ = std::make_shared<cv::Mat>(
                cv_bridge::toCvShare(msg, "bgr8")->image);
            right_timestamp_ = msg->header.stamp;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error converting right image: %s", e.what());
        }
    }
    
    void process_images()
    {
        // Frame skipping for performance
        int skip_frames = this->get_parameter("skip_frames").as_int();
        frame_skip_counter_++;
        if (frame_skip_counter_ < skip_frames) {
            return;
        }
        frame_skip_counter_ = 0;
        
        // Check if we have both images
        std::shared_ptr<cv::Mat> left_img, right_img;
        rclcpp::Time current_timestamp;
        
        {
            std::lock_guard<std::mutex> lock(image_lock_);
            if (!current_left_ || !current_right_) {
                return;
            }
            left_img = current_left_;
            right_img = current_right_;
            current_timestamp = left_timestamp_;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool pub_undistort_images = this->get_parameter("pub_undistort_images").as_bool();
        bool pub_disparity_map = this->get_parameter("pub_disparity_map").as_bool();
        bool pub_depth_map = this->get_parameter("pub_depth_map").as_bool();
        bool show_images = this->get_parameter("show_images").as_bool();
        bool use_wls_filter = this->get_parameter("use_wls_filter").as_bool();
        
        // Fast undistortion using pre-computed maps
        cv::Mat left_undist, right_undist;
        undistort_images_fast(*left_img, *right_img, left_undist, right_undist);
        
        if (!left_undist.empty() && !right_undist.empty()) {
            // Convert to grayscale efficiently
            cv::Mat left_gray, right_gray;
            cv::cvtColor(left_undist, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_undist, right_gray, cv::COLOR_BGR2GRAY);
            
            // Compute disparity with or without WLS filter
            cv::Mat disparity;
            if (use_wls_filter && wls_filter_) {
                disparity = compute_disparity_with_wls(left_gray, right_gray);
            } else {
                cv::Mat disp_raw;
                stereo_matcher_left_->compute(left_gray, right_gray, disp_raw);
                disp_raw.convertTo(disparity, CV_32F, 1.0/16.0);
            }
            
            // Compute depth map
            cv::Mat depth_map = compute_depth_map(disparity);
            
            if (pub_undistort_images) {
                publish_undistorted_images_fast(left_undist, right_undist, current_timestamp);
            }
            
            if (pub_disparity_map) {
                publish_disparity_images_fast(disparity, current_timestamp);
            }
            
            if (pub_depth_map) {
                publish_depth_images_fast(depth_map, current_timestamp);
            }
            
            if (show_images) {
                if (process_count_ % 3 == 0) {
                    display_images_fast(left_undist, right_undist, depth_map, left_gray, right_gray);
                }
            }
        }
        
        // Performance monitoring
        process_count_++;
        if (process_count_ % 30 == 0) {
            auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            RCLCPP_INFO(this->get_logger(), "Processing time: %ld ms", processing_time);
        }
    }
    
    cv::Mat compute_disparity_with_wls(const cv::Mat& left_gray, const cv::Mat& right_gray)
    {
        try {
            cv::Mat disp_left, disp_right;
            
            // Compute left disparity
            stereo_matcher_left_->compute(left_gray, right_gray, disp_left);
            
            // Compute right disparity for consistency check
            stereo_matcher_right_->compute(right_gray, left_gray, disp_right);
            
            // Apply WLS filter
            cv::Mat filtered_disp;
            wls_filter_->filter(disp_left, left_gray, filtered_disp, disp_right);
            
            // Convert to float32 and scale
            cv::Mat result;
            filtered_disp.convertTo(result, CV_32F, 1.0/16.0);
            
            return result;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in WLS disparity computation: %s", e.what());
            // Fallback to regular disparity
            cv::Mat disp_raw, result;
            stereo_matcher_left_->compute(left_gray, right_gray, disp_raw);
            disp_raw.convertTo(result, CV_32F, 1.0/16.0);
            return result;
        }
    }
    
    void undistort_images_fast(const cv::Mat& left_img, const cv::Mat& right_img,
                              cv::Mat& left_undist, cv::Mat& right_undist)
    {
        if (left_map1_.empty() || left_map2_.empty() || 
            right_map1_.empty() || right_map2_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Undistortion maps are empty");
            left_undist = left_img.clone();
            right_undist = right_img.clone();
            return;
        }
        
        // Fast remapping
        cv::remap(left_img, left_undist, left_map1_, left_map2_, 
                 cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        cv::remap(right_img, right_undist, right_map1_, right_map2_, 
                 cv::INTER_NEAREST, cv::BORDER_CONSTANT);
    }
    
    void publish_undistorted_images_fast(const cv::Mat& left_undist, const cv::Mat& right_undist,
                                        const rclcpp::Time& timestamp)
    {
        try {
            // Create messages efficiently
            auto left_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", left_undist).toImageMsg();
            auto right_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", right_undist).toImageMsg();
            
            // Use original timestamp for synchronization
            left_msg->header.stamp = timestamp;
            right_msg->header.stamp = timestamp;
            left_msg->header.frame_id = "cam0_undis";
            right_msg->header.frame_id = "cam1_undis";
            
            // Publish
            left_undist_pub_->publish(*left_msg);
            right_undist_pub_->publish(*right_msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing undistorted images: %s", e.what());
        }
    }
    
    void publish_depth_images_fast(const cv::Mat& depth_map, const rclcpp::Time& timestamp)
    {
        try {
            // Normalize for visualization
            cv::Mat depth_normalized, depth_uint8, depth_colormap;
            cv::normalize(depth_map, depth_normalized, 0, 255, cv::NORM_MINMAX);
            depth_normalized.convertTo(depth_uint8, CV_8U);
            cv::applyColorMap(depth_uint8, depth_colormap, cv::COLORMAP_RAINBOW);
            
            // Get image dimensions
            int h = depth_map.rows;
            int w = depth_map.cols;
            
            // Define key points and overlay text
            std::vector<std::pair<std::string, cv::Point>> keypoints = {
                {"Center", cv::Point(w/2, h/2)},
                {"Left", cv::Point(w/4, h/2)},
                {"Right", cv::Point(3*w/4, h/2)},
                {"Top-Left", cv::Point(160, 0)},
                {"Bottom-Right", cv::Point(w-400, h-100)}
            };
            
            for (const auto& kp : keypoints) {
                if (kp.second.x >= 0 && kp.second.x < w && kp.second.y >= 0 && kp.second.y < h) {
                    float depth_value = depth_map.at<float>(kp.second.y, kp.second.x);
                    std::string text = kp.first + ": " + std::to_string(depth_value).substr(0, 4) + " m";
                    cv::putText(depth_colormap, text, cv::Point(kp.second.x + 5, kp.second.y + 25),
                               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                }
            }
            
            // Create ROS message
            auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", depth_colormap).toImageMsg();
            depth_msg->header.stamp = timestamp;
            depth_msg->header.frame_id = "cam0_depth";
            
            // Publish
            depth_pub_->publish(*depth_msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing depth image with annotations: %s", e.what());
        }
    }
    
    void publish_disparity_images_fast(const cv::Mat& disparity, const rclcpp::Time& timestamp)
    {
        try {
            // Handle potential invalid disparity values
            cv::Mat valid_mask = disparity > 0;
            if (cv::countNonZero(valid_mask) == 0) {
                RCLCPP_WARN(this->get_logger(), "No valid disparity values found");
                return;
            }
            
            double min_val, max_val;
            cv::minMaxLoc(disparity, &min_val, &max_val, nullptr, nullptr, valid_mask);
            
            cv::Mat disparity_norm = disparity / max_val;
            cv::Mat disparity_invert = 1.0 - disparity_norm;
            
            cv::Mat disparity_uint8, disp_colormap;
            disparity_invert.convertTo(disparity_uint8, CV_8U, 255.0);
            cv::applyColorMap(disparity_uint8, disp_colormap, cv::COLORMAP_RAINBOW);
            
            // Create message
            auto disparity_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", disp_colormap).toImageMsg();
            disparity_msg->header.stamp = timestamp;
            disparity_msg->header.frame_id = "cam0_disparity";
            
            // Publish
            disparity_pub_->publish(*disparity_msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing disparity images: %s", e.what());
        }
    }
    
    void display_images_fast(const cv::Mat& left_undist, const cv::Mat& right_undist,
                            const cv::Mat& depth_map, const cv::Mat& left_gray, const cv::Mat& right_gray)
    {
        try {
            // Display depth and disparity with reduced frequency
            if (process_count_ % 5 == 0) {
                display_depth_map_fast(depth_map);
                
                // Use the current disparity method
                bool use_wls_filter = this->get_parameter("use_wls_filter").as_bool();
                cv::Mat disparity;
                if (use_wls_filter && wls_filter_) {
                    disparity = compute_disparity_with_wls(left_gray, right_gray);
                } else {
                    cv::Mat disp_raw;
                    stereo_matcher_left_->compute(left_gray, right_gray, disp_raw);
                    disp_raw.convertTo(disparity, CV_32F, 1.0/16.0);
                }
                
                display_disparity_map_fast(disparity);
            }
            
            cv::waitKey(1);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in display: %s", e.what());
        }
    }
    
    void display_depth_map_fast(const cv::Mat& depth_map)
    {
        try {
            cv::Mat depth_normalized, depth_uint8, depth_colormap;
            cv::normalize(depth_map, depth_normalized, 0, 255, cv::NORM_MINMAX);
            depth_normalized.convertTo(depth_uint8, CV_8U);
            cv::applyColorMap(depth_uint8, depth_colormap, cv::COLORMAP_RAINBOW);
            
            int display_width = static_cast<int>(depth_colormap.cols * display_scale_);
            int display_height = static_cast<int>(depth_colormap.rows * display_scale_);
            
            cv::Mat depth_small;
            cv::resize(depth_colormap, depth_small, cv::Size(display_width, display_height), 
                      0, 0, cv::INTER_AREA);
            
            cv::imshow("Depth Map", depth_small);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error displaying depth map: %s", e.what());
        }
    }
    
    void display_disparity_map_fast(const cv::Mat& disparity)
    {
        try {
            cv::Mat valid_mask = disparity > 0;
            if (cv::countNonZero(valid_mask) == 0) {
                return;
            }
            
            double disp_min, disp_max;
            cv::minMaxLoc(disparity, &disp_min, &disp_max, nullptr, nullptr, valid_mask);
            
            if (disp_max > disp_min) {
                cv::Mat disparity_norm = (disparity - disp_min) / (disp_max - disp_min);
                disparity_norm = 1.0 - disparity_norm;
                
                cv::Mat disparity_uint8, disparity_colormap;
                disparity_norm.convertTo(disparity_uint8, CV_8U, 255.0);
                cv::applyColorMap(disparity_uint8, disparity_colormap, cv::COLORMAP_RAINBOW);
                
                int display_width = static_cast<int>(disparity_colormap.cols * display_scale_);
                int display_height = static_cast<int>(disparity_colormap.rows * display_scale_);
                
                cv::Mat disp_small;
                cv::resize(disparity_colormap, disp_small, cv::Size(display_width, display_height), 
                          0, 0, cv::INTER_AREA);
                
                cv::imshow("Disparity Map", disp_small);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error displaying disparity map: %s", e.what());
        }
    }
    
    void generate_undistortion_maps(const cv::Size& img_shape)
    {
        try {
            // Generate maps with optimal data type for performance
            cv::fisheye::initUndistortRectifyMap(
                left_K_, left_D_, cv::Mat::eye(3, 3, CV_64F), left_K_, 
                img_shape, CV_16SC2, left_map1_, left_map2_);
            
            cv::fisheye::initUndistortRectifyMap(
                right_K_, right_D_, cv::Mat::eye(3, 3, CV_64F), right_K_, 
                img_shape, CV_16SC2, right_map1_, right_map2_);
            
            // Convert left_K_ to string and print it
            std::ostringstream oss;
            oss << "left_K_:\n" << left_K_;
            RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());         
               
            RCLCPP_INFO(this->get_logger(), "Generated optimized undistortion maps for both cameras");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error generating undistortion maps: %s", e.what());
        }
    }
    
    void path_calibration()
    {
        try {
            std::string pkg_name = "robot_camera";
            std::string path_pkg_share_path = ament_index_cpp::get_package_share_directory(pkg_name);
            
            // Extract workspace path
            size_t install_pos = path_pkg_share_path.find("install");
            std::string ws_path = path_pkg_share_path.substr(0, install_pos);
            
            std::string file_name_yaml = this->get_parameter("file_name_yaml").as_string();
            path_yaml_ = ws_path + "src/" + pkg_name + "/config/" + file_name_yaml;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error setting calibration path: %s", e.what());
        }
    }
    
    void load_calibration()
    {
        try {
            if (std::filesystem::exists(path_yaml_)) {
                YAML::Node calib_data = YAML::LoadFile(path_yaml_);
                
                // Load calibration matrices
                auto left_K_seq = calib_data["left_K"].as<std::vector<double>>();
                auto left_D_seq = calib_data["left_D"].as<std::vector<double>>();
                auto right_K_seq = calib_data["right_K"].as<std::vector<double>>();
                auto right_D_seq = calib_data["right_D"].as<std::vector<double>>();
                auto R_seq = calib_data["R"].as<std::vector<double>>();
                auto T_seq = calib_data["T"].as<std::vector<double>>();
                
                // Convert to OpenCV matrices
                left_K_ = cv::Mat(3, 3, CV_64F, left_K_seq.data()).clone();
                left_D_ = cv::Mat(4, 1, CV_64F, left_D_seq.data()).clone();
                right_K_ = cv::Mat(3, 3, CV_64F, right_K_seq.data()).clone();
                right_D_ = cv::Mat(4, 1, CV_64F, right_D_seq.data()).clone();
                R_ = cv::Mat(3, 3, CV_64F, R_seq.data()).clone();
                T_ = cv::Mat(3, 1, CV_64F, T_seq.data()).clone();
                
                // Get image shape
                auto img_shape_seq = calib_data["image_shape"].as<std::vector<int>>();
                cv::Size img_shape(img_shape_seq[1], img_shape_seq[0]); // width, height
                
                generate_undistortion_maps(img_shape);
                
                RCLCPP_INFO(this->get_logger(), "Loaded existing calibration and generated maps");
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration: %s", e.what());
        }
    }
    
    void setup_stereo_matcher()
    {
        try {
            int max_disp = this->get_parameter("max_disparities").as_int();
            int min_disp = this->get_parameter("min_disparities").as_int();
            int block_size = this->get_parameter("block_size").as_int();
            int uniqueness_ratio = this->get_parameter("uniqueness_ratio").as_int();
            int speckle_window_size = this->get_parameter("speckle_window_size").as_int();
            int speckle_range = this->get_parameter("speckle_range").as_int();
            int pre_filter_cap = this->get_parameter("pre_filter_cap").as_int();
            
            // Create left stereo matcher (main matcher)
            stereo_matcher_left_ = cv::StereoSGBM::create(
                min_disp,                           // minDisparity
                max_disp,                           // numDisparities
                block_size,                         // blockSize
                8 * 3 * block_size * block_size,    // P1
                32 * 3 * block_size * block_size,   // P2
                1,                                  // disp12MaxDiff
                pre_filter_cap,                     // preFilterCap
                uniqueness_ratio,                   // uniquenessRatio
                speckle_window_size,                // speckleWindowSize
                speckle_range,                      // speckleRange
                cv::StereoSGBM::MODE_SGBM_3WAY     // mode
            );
            
            // Create right stereo matcher for WLS filter
            stereo_matcher_right_ = cv::ximgproc::createRightMatcher(stereo_matcher_left_);
            
            // Create WLS filter
            double wls_lambda = this->get_parameter("wls_lambda").as_double();
            double wls_sigma = this->get_parameter("wls_sigma").as_double();
            
            wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_matcher_left_);
            wls_filter_->setLambda(wls_lambda);
            wls_filter_->setSigmaColor(wls_sigma);
            
            RCLCPP_INFO(this->get_logger(), 
                       "Initialized stereo matchers and WLS filter (lambda=%.1f, sigma=%.1f)", 
                       wls_lambda, wls_sigma);
                       
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error setting up stereo matcher: %s", e.what());
        }
    }
    
    cv::Mat compute_depth_map(const cv::Mat& disparity)
    {
        try {
            // Convert disparity to depth efficiently
            double fx = left_K_.at<double>(0, 0);
            double baseline = std::abs(T_.at<double>(0, 0));
            
            // Handle invalid disparity values
            cv::Mat disparity_fixed;
            cv::Mat mask = disparity <= 0;
            disparity.copyTo(disparity_fixed);
            disparity_fixed.setTo(0.1, mask);
            
            cv::Mat depth_map = fx * baseline / disparity_fixed;
            
            // Mark invalid pixels
            depth_map.setTo(0, mask);
            
            // Apply depth limits
            double max_depth = this->get_parameter("max_depth").as_double();
            double min_depth = this->get_parameter("min_depth").as_double();
            
            cv::Mat depth_clamped;
            cv::max(depth_map, min_depth, depth_clamped);
            cv::min(depth_clamped, max_depth, depth_map);
            
            return depth_map;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error computing depth map: %s", e.what());
            return cv::Mat();
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Stereo2Depth>();
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception in main: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}