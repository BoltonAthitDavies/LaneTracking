#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <memory>

class StereoUndistortNode : public rclcpp::Node
{
public:
    StereoUndistortNode() : Node("stereo_undistort_node")
    {
        // Declare parameters
        this->declare_parameter<std::string>("file_name_yaml", "real_calibration_v2.yaml");
        
        // Get calibration file path
        path_calibration();
        
        // Load calibration parameters
        if (!load_calibration()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration parameters");
            return;
        }
        
        // Create QoS profile with BEST_EFFORT reliability
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        // Create subscribers
        left_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/cam0/image_raw", qos,
            std::bind(&StereoUndistortNode::left_image_callback, this, std::placeholders::_1));
            
        right_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/cam1/image_raw", qos,
            std::bind(&StereoUndistortNode::right_image_callback, this, std::placeholders::_1));
        
        // Create publishers for undistorted images
        left_undistorted_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/cam0/image_undistorted", 1);
        right_undistorted_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/cam1/image_undistorted", 1);
        
        RCLCPP_INFO(this->get_logger(), "Stereo undistortion node initialized");
    }

private:
    void path_calibration()
    {
        try {
            std::string pkg_name = "robot_camera";
            std::string pkg_share_path = ament_index_cpp::get_package_share_directory(pkg_name);
            
            // Extract workspace path
            std::filesystem::path pkg_path(pkg_share_path);
            std::string ws_path;
            
            // Navigate up to find workspace root (before install directory)
            for (auto& part : pkg_path) {
                if (part == "install") {
                    break;
                }
                ws_path = ws_path.empty() ? part.string() : ws_path + "/" + part.string();
            }
            
            std::string file_name_yaml = this->get_parameter("file_name_yaml").as_string();
            path_yaml_ = ws_path + "/src/" + pkg_name + "/config/" + file_name_yaml;
            
            RCLCPP_INFO(this->get_logger(), "Calibration file path: %s", path_yaml_.c_str());
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error setting calibration path: %s", e.what());
        }
    }
    
    bool load_calibration()
    {
        try {
            if (!std::filesystem::exists(path_yaml_)) {
                RCLCPP_ERROR(this->get_logger(), "Calibration file does not exist: %s", path_yaml_.c_str());
                return false;
            }
            
            YAML::Node calib_data = YAML::LoadFile(path_yaml_);
            
            // Load left camera parameters (handle 2D array format)
            auto left_K_seq = calib_data["left_K"];
            auto left_D_seq = calib_data["left_D"];
            
            // Extract K matrix values (3x3 matrix in row-major order)
            left_fx_ = left_K_seq[0][0].as<double>();  // K[0,0]
            left_fy_ = left_K_seq[1][1].as<double>();  // K[1,1]
            left_cx_ = left_K_seq[0][2].as<double>();  // K[0,2]
            left_cy_ = left_K_seq[1][2].as<double>();  // K[1,2]
            
            // Extract distortion coefficients (handle 2D array format)
            left_k1_ = left_D_seq[0][0].as<double>();
            left_k2_ = left_D_seq[0][1].as<double>();
            left_p1_ = left_D_seq[0][2].as<double>();
            left_p2_ = left_D_seq[0][3].as<double>();
            
            // Load right camera parameters (handle 2D array format)
            auto right_K_seq = calib_data["right_K"];
            auto right_D_seq = calib_data["right_D"];
            
            // Extract K matrix values (3x3 matrix in row-major order)
            right_fx_ = right_K_seq[0][0].as<double>();  // K[0,0]
            right_fy_ = right_K_seq[1][1].as<double>();  // K[1,1]
            right_cx_ = right_K_seq[0][2].as<double>();  // K[0,2]
            right_cy_ = right_K_seq[1][2].as<double>();  // K[1,2]
            
            // Extract distortion coefficients (handle 2D array format)
            right_k1_ = right_D_seq[0][0].as<double>();
            right_k2_ = right_D_seq[0][1].as<double>();
            right_p1_ = right_D_seq[0][2].as<double>();
            right_p2_ = right_D_seq[0][3].as<double>();
            
            // Load image shape
            auto img_shape = calib_data["image_shape"];
            image_height_ = img_shape[0].as<int>();
            image_width_ = img_shape[1].as<int>();
            
            RCLCPP_INFO(this->get_logger(), "Loaded calibration parameters successfully");
            RCLCPP_INFO(this->get_logger(), "Image size: %dx%d", image_width_, image_height_);
            RCLCPP_INFO(this->get_logger(), "Left camera - fx: %.3f, fy: %.3f, cx: %.3f, cy: %.3f", 
                       left_fx_, left_fy_, left_cx_, left_cy_);
            RCLCPP_INFO(this->get_logger(), "Left camera distortion - k1: %.6f, k2: %.6f, p1: %.6f, p2: %.6f", 
                       left_k1_, left_k2_, left_p1_, left_p2_);
            RCLCPP_INFO(this->get_logger(), "Right camera - fx: %.3f, fy: %.3f, cx: %.3f, cy: %.3f", 
                       right_fx_, right_fy_, right_cx_, right_cy_);
            RCLCPP_INFO(this->get_logger(), "Right camera distortion - k1: %.6f, k2: %.6f, p1: %.6f, p2: %.6f", 
                       right_k1_, right_k2_, right_p1_, right_p2_);
            
            return true;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load calibration: %s", e.what());
            return false;
        }
    }
    
    cv::Mat undistort_image(const cv::Mat& distorted_image, bool is_left_camera)
    {
        int rows = distorted_image.rows;
        int cols = distorted_image.cols;
        cv::Mat undistorted_image = cv::Mat::zeros(rows, cols, distorted_image.type());
        
        // Select camera parameters
        double fx, fy, cx, cy, k1, k2, p1, p2;
        if (is_left_camera) {
            fx = left_fx_; fy = left_fy_; cx = left_cx_; cy = left_cy_;
            k1 = left_k1_; k2 = left_k2_; p1 = left_p1_; p2 = left_p2_;
        } else {
            fx = right_fx_; fy = right_fy_; cx = right_cx_; cy = right_cy_;
            k1 = right_k1_; k2 = right_k2_; p1 = right_p1_; p2 = right_p2_;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Undistorting %s image: %dx%d, fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
                    is_left_camera ? "left" : "right", cols, rows, fx, fy, cx, cy);
        
        int valid_pixels = 0;
        int total_pixels = rows * cols;
        
        // Manual undistortion implementation (based on your template)
        for (int v = 0; v < rows; v++) {
            for (int u = 0; u < cols; u++) {
                // Convert pixel coordinates to normalized coordinates
                double x = (u - cx) / fx;
                double y = (v - cy) / fy;
                double r = sqrt(x * x + y * y);
                
                // Apply distortion model
                double radial_distortion = 1 + k1 * r * r + k2 * r * r * r * r;
                double x_distorted = x * radial_distortion + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
                double y_distorted = y * radial_distortion + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
                
                // Convert back to pixel coordinates
                double u_distorted = fx * x_distorted + cx;
                double v_distorted = fy * y_distorted + cy;
                
                // Bounds checking with margin
                if (u_distorted >= 0 && v_distorted >= 0 && 
                    u_distorted < (cols - 1) && v_distorted < (rows - 1)) {
                    
                    int u_d = static_cast<int>(round(u_distorted));
                    int v_d = static_cast<int>(round(v_distorted));
                    
                    // Double check bounds after rounding
                    if (u_d >= 0 && v_d >= 0 && u_d < cols && v_d < rows) {
                        if (distorted_image.channels() == 1) {
                            // Grayscale image
                            undistorted_image.at<uchar>(v, u) = distorted_image.at<uchar>(v_d, u_d);
                        } else if (distorted_image.channels() == 3) {
                            // Color image (BGR)
                            undistorted_image.at<cv::Vec3b>(v, u) = distorted_image.at<cv::Vec3b>(v_d, u_d);
                        }
                        valid_pixels++;
                    }
                }
            }
        }
        
        double valid_ratio = (double)valid_pixels / total_pixels;
        RCLCPP_DEBUG(this->get_logger(), "Undistortion completed: %.1f%% valid pixels (%d/%d)", 
                    valid_ratio * 100, valid_pixels, total_pixels);
        
        if (valid_ratio < 0.5) {
            RCLCPP_WARN(this->get_logger(), "Warning: Only %.1f%% of pixels were successfully undistorted. "
                       "Check calibration parameters.", valid_ratio * 100);
        }
        
        return undistorted_image;
    }
    
    void left_image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            RCLCPP_DEBUG(this->get_logger(), "Received left image: %dx%d, encoding: %s", 
                        msg->width, msg->height, msg->encoding.c_str());
            
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // Check if image is valid
            if (cv_ptr->image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Received empty left image");
                return;
            }
            
            // Log some image statistics
            cv::Scalar mean_val = cv::mean(cv_ptr->image);
            RCLCPP_DEBUG(this->get_logger(), "Left image mean: [%.1f, %.1f, %.1f]", 
                        mean_val[0], mean_val[1], mean_val[2]);
            
            // Undistort the image
            cv::Mat undistorted = undistort_image(cv_ptr->image, true);
            
            // Check undistorted image
            if (undistorted.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Undistorted left image is empty");
                return;
            }
            
            cv::Scalar undist_mean = cv::mean(undistorted);
            RCLCPP_DEBUG(this->get_logger(), "Left undistorted mean: [%.1f, %.1f, %.1f]", 
                        undist_mean[0], undist_mean[1], undist_mean[2]);
            
            // Convert back to ROS message
            cv_bridge::CvImage undistorted_msg;
            undistorted_msg.header = msg->header;
            undistorted_msg.encoding = msg->encoding;
            undistorted_msg.image = undistorted;
            
            // Publish undistorted image
            left_undistorted_pub_->publish(*undistorted_msg.toImageMsg());
            
            RCLCPP_DEBUG(this->get_logger(), "Left image undistorted and published");
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception for left image: %s", e.what());
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing left image: %s", e.what());
        }
    }
    
    void right_image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            RCLCPP_DEBUG(this->get_logger(), "Received right image: %dx%d, encoding: %s", 
                        msg->width, msg->height, msg->encoding.c_str());
            
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // Check if image is valid
            if (cv_ptr->image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Received empty right image");
                return;
            }
            
            // Log some image statistics
            cv::Scalar mean_val = cv::mean(cv_ptr->image);
            RCLCPP_DEBUG(this->get_logger(), "Right image mean: [%.1f, %.1f, %.1f]", 
                        mean_val[0], mean_val[1], mean_val[2]);
            
            // Undistort the image
            cv::Mat undistorted = undistort_image(cv_ptr->image, false);
            
            // Check undistorted image
            if (undistorted.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Undistorted right image is empty");
                return;
            }
            
            cv::Scalar undist_mean = cv::mean(undistorted);
            RCLCPP_DEBUG(this->get_logger(), "Right undistorted mean: [%.1f, %.1f, %.1f]", 
                        undist_mean[0], undist_mean[1], undist_mean[2]);
            
            // Convert back to ROS message
            cv_bridge::CvImage undistorted_msg;
            undistorted_msg.header = msg->header;
            undistorted_msg.encoding = msg->encoding;
            undistorted_msg.image = undistorted;
            
            // Publish undistorted image
            right_undistorted_pub_->publish(*undistorted_msg.toImageMsg());
            
            RCLCPP_DEBUG(this->get_logger(), "Right image undistorted and published");
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception for right image: %s", e.what());
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing right image: %s", e.what());
        }
    }
    
    // ROS2 subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_undistorted_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_undistorted_pub_;
    
    // Calibration parameters
    std::string path_yaml_;
    int image_width_, image_height_;
    
    // Left camera parameters
    double left_fx_, left_fy_, left_cx_, left_cy_;
    double left_k1_, left_k2_, left_p1_, left_p2_;
    
    // Right camera parameters
    double right_fx_, right_fy_, right_cx_, right_cy_;
    double right_k1_, right_k2_, right_p1_, right_p2_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<StereoUndistortNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting stereo undistortion node...");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}