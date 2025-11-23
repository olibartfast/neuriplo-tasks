#include "vision-core/optical_flow/raft_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float { 
        return static_cast<float>(arg); 
    }, elem);
}

} // anonymous namespace

cv::Mat RaftPostprocessor::make_color_wheel() {
    // Constants for color wheel segments
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    
    // Red to Yellow
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 255 * i / RY, 0);
    }
    
    // Yellow to Green
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 - 255 * i / YG, 255, 0);
    }
    
    // Green to Cyan
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255, 255 * i / GC);
    }
    
    // Cyan to Blue
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255 - 255 * i / CB, 255);
    }
    
    // Blue to Magenta
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 * i / BM, 0, 255);
    }
    
    // Magenta to Red
    for (int i = 0; i < MR; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 0, 255 - 255 * i / MR);
    }
    
    return colorwheel;
}

cv::Mat RaftPostprocessor::flow_to_color(const cv::Mat& flow_mat) {
    cv::Mat flow_parts[2];
    cv::split(flow_mat, flow_parts);
    cv::Mat u = flow_parts[0], v = flow_parts[1];

    // Compute magnitude and angle
    cv::Mat magnitude, angle;
    cv::cartToPolar(u, v, magnitude, angle);

    // Normalize magnitude
    double mag_max;
    cv::minMaxLoc(magnitude, nullptr, &mag_max);
    if (mag_max > 0) {
        magnitude /= mag_max;
    }

    // Convert angle to [0, 1] range
    angle *= (1.0 / (2.0 * CV_PI));

    // Apply color wheel
    cv::Mat colorwheel = make_color_wheel();
    const int ncols = colorwheel.rows;
    cv::Mat flow_color(flow_mat.size(), CV_8UC3);

    for (int i = 0; i < flow_mat.rows; ++i) {
        for (int j = 0; j < flow_mat.cols; ++j) {
            float mag = magnitude.at<float>(i, j);
            float ang = angle.at<float>(i, j);

            // Find nearest colors in the wheel
            int k0 = static_cast<int>(ang * (ncols - 1));
            int k1 = (k0 + 1) % ncols;
            float f = (ang * (ncols - 1)) - k0;

            // Get colors from the wheel
            cv::Vec3b col0 = colorwheel.at<cv::Vec3b>(k0);
            cv::Vec3b col1 = colorwheel.at<cv::Vec3b>(k1);

            // Interpolate colors
            cv::Vec3b color;
            for (int ch = 0; ch < 3; ++ch) {
                float channel_value = (1 - f) * col0[ch] + f * col1[ch];
                // Apply magnitude modulation
                if (mag <= 1) {
                    channel_value = 255 - mag * (255 - channel_value);
                } else {
                    channel_value *= 0.75;
                }
                color[ch] = static_cast<uchar>(channel_value);
            }

            flow_color.at<cv::Vec3b>(i, j) = color;
        }
    }

    return flow_color;
}

OpticalFlowResult RaftPostprocessor::postprocess(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size)
{
    if (shape.size() != 4 || shape[1] != 2) {
        throw std::invalid_argument("RAFT output shape must be [batch, 2, H, W]");
    }

    const int64_t height = shape[2];
    const int64_t width = shape[3];
    [[maybe_unused]] const size_t expected_size = height * width * 2;

    // Channel offsets (NCHW format: u channel, then v channel)
    const int64_t u_channel_offset = 0;
    const int64_t v_channel_offset = height * width;

    // Create flow matrix
    cv::Mat flow_mat(height, width, CV_32FC2);
    float* flow_ptr = flow_mat.ptr<float>();

    // Reconstruct the flow field
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            // U channel (horizontal flow)
            flow_ptr[y * width * 2 + x * 2] = 
                get_float(output[u_channel_offset + y * width + x]);
            // V channel (vertical flow)
            flow_ptr[y * width * 2 + x * 2 + 1] = 
                get_float(output[v_channel_offset + y * width + x]);
        }
    }

    // Calculate maximum displacement
    cv::Mat magnitude, angle;
    std::vector<cv::Mat> flow_parts;
    cv::split(flow_mat, flow_parts);
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);
    
    double max_displacement;
    cv::minMaxLoc(magnitude, nullptr, &max_displacement);

    // Create colored visualization
    cv::Mat flow_color = flow_to_color(flow_mat);

    // Resize if necessary
    cv::Mat final_flow = flow_mat;
    cv::Mat final_color = flow_color;
    
    if (frame_size != cv::Size(width, height)) {
        cv::resize(flow_mat, final_flow, frame_size, 0, 0, cv::INTER_LINEAR);
        cv::resize(flow_color, final_color, frame_size, 0, 0, cv::INTER_LINEAR);
        
        // Scale flow values proportionally when resizing
        float scale_x = static_cast<float>(frame_size.width) / width;
        float scale_y = static_cast<float>(frame_size.height) / height;
        
        std::vector<cv::Mat> channels;
        cv::split(final_flow, channels);
        channels[0] *= scale_x;
        channels[1] *= scale_y;
        cv::merge(channels, final_flow);
    }

    return OpticalFlowResult(final_flow, final_color, max_displacement);
}

} // namespace vision_core
