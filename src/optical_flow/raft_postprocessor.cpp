#include "vision-core/optical_flow/raft_postprocessor.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace vision_core {

RaftPostprocessor::RaftPostprocessor() {}

std::vector<OpticalFlow> RaftPostprocessor::postprocess(
    const std::vector<TensorElement>& flow_output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    if (flow_output.empty() || shape.empty()) {
        return {};
    }

    // RAFT output: [1, 2, H, W] (dx, dy)
    if (shape.size() < 4) return {};
    
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    if (channels != 2) return {};
    
    // Debug: Check what type we actually have
    std::cout << "First element type index: " << flow_output[0].index() << std::endl;
    if (std::holds_alternative<float>(flow_output[0])) {
        std::cout << "Data is float type" << std::endl;
    } else if (std::holds_alternative<int>(flow_output[0])) {
        std::cout << "Data is int type" << std::endl;
    } else {
        std::cout << "Data is other type" << std::endl;
    }
    
    // Robust tensor data access - handle float tensors  
    const float* data = std::get_if<float>(&flow_output[0]);
    if (!data) {
        std::cout << "Converting non-float tensor data" << std::endl;
        // Fallback: convert other types to float
        static std::vector<float> temp_buffer;
        temp_buffer.resize(flow_output.size());
        for (size_t i = 0; i < flow_output.size(); ++i) {
            temp_buffer[i] = getTensorFloat(flow_output[i]);
        }
        data = temp_buffer.data();
        std::cout << "First converted values: ";
        for (int i = 0; i < std::min(5, (int)temp_buffer.size()); ++i) {
            std::cout << temp_buffer[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Using direct float data, first values: ";
        for (int i = 0; i < std::min(5, (int)flow_output.size()); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Create flow matrix [H, W, 2] like master branch
    cv::Mat flow(height, width, CV_32FC2);
    float* flow_ptr = reinterpret_cast<float*>(flow.data);
    
    // Try different tensor interpretations to find the right one
    std::cout << "Trying different tensor layouts..." << std::endl;
    
    // Interpretation 1: Original approach
    const int u_channel_offset = 0;
    const int v_channel_offset = height * width;
    
    std::cout << "Testing interpretation 1: u_offset=0, v_offset=" << v_channel_offset << std::endl;
    std::cout << "Sample values - U: " << data[0] << ", " << data[1] << ", " << data[2] << std::endl;
    std::cout << "Sample values - V: " << data[v_channel_offset] << ", " << data[v_channel_offset+1] << ", " << data[v_channel_offset+2] << std::endl;
    
    // Find where the significant flow values are located
    std::cout << "Scanning tensor for significant values..." << std::endl;
    int total_size = height * width * 2;
    for (int i = 0; i < total_size && i < 100; i++) {
        if (std::abs(data[i]) > 0.1f) {
            std::cout << "Significant value at index " << i << ": " << data[i] << std::endl;
        }
    }
    
    // Try alternative tensor layout: interleaved channels [u,v,u,v,u,v...]
    std::cout << "Testing interleaved layout..." << std::endl;
    std::cout << "Interleaved samples: " << data[0] << "(u), " << data[1] << "(v), " << data[2] << "(u), " << data[3] << "(v)" << std::endl;
    
    // Let's try the exact pattern from master branch
    // Since we know there are significant values (max magnitude 4.3383), they must be somewhere
    
    // Try approach 1: Master branch style with std::get<float> access
    std::cout << "Trying to access tensor data like master branch..." << std::endl;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Master branch pattern: direct TensorElement access
            flow_ptr[y * width * 2 + x * 2] = getTensorFloat(flow_output[u_channel_offset + y * width + x]);
            flow_ptr[y * width * 2 + x * 2 + 1] = getTensorFloat(flow_output[v_channel_offset + y * width + x]);
        }
    }
    
    // Debug: Check what we actually got in the flow matrix
    double flow_min, flow_max;
    cv::Mat flow_flat = flow.reshape(1, flow.total());  // Flatten to 1D
    cv::minMaxLoc(flow_flat, &flow_min, &flow_max);
    std::cout << "Reconstructed flow matrix range: " << flow_min << " to " << flow_max << std::endl;
    
    // Resize to original frame size if needed (with proper flow scaling)
    if (frame_size.width != width || frame_size.height != height) {
        cv::Mat resized_flow;
        cv::resize(flow, resized_flow, frame_size);
        
        // Scale flow values proportionally
        std::vector<cv::Mat> flow_channels;
        cv::split(resized_flow, flow_channels);
        flow_channels[0] *= static_cast<float>(frame_size.width) / width;   // Scale U
        flow_channels[1] *= static_cast<float>(frame_size.height) / height; // Scale V
        cv::merge(flow_channels, flow);
    }
    
    // Split flow into separate U and V components for visualization
    std::vector<cv::Mat> flow_channels;
    cv::split(flow, flow_channels);
    cv::Mat flow_u = flow_channels[0];
    cv::Mat flow_v = flow_channels[1];
    
    OpticalFlow result;
    result.raw_flow = flow.clone();
    
    // Calculate magnitude and max displacement
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_u, flow_v, magnitude, angle);
    double max_disp;
    cv::minMaxLoc(magnitude, nullptr, &max_disp);
    result.max_displacement = static_cast<float>(max_disp);
    
    // Create color visualization using master branch approach
    result.flow = visualizeFlow(flow_u, flow_v);
    
    return {result};
}

float RaftPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

cv::Mat RaftPostprocessor::makeColorwheel() {
    // Constants for color wheel (matching master branch exactly)
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    // RY - Red to Yellow (BGR format: Blue=255, Green=increasing, Red=0)
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 255 * i / RY, 0);
    }
    // YG - Yellow to Green (BGR format: Blue=255-decreasing, Green=255, Red=0)
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 - 255 * i / YG, 255, 0);
    }
    // GC - Green to Cyan (BGR format: Blue=0, Green=255, Red=increasing)
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255, 255 * i / GC);
    }
    // CB - Cyan to Blue (BGR format: Blue=0, Green=255-decreasing, Red=255)
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255 - 255 * i / CB, 255);
    }
    // BM - Blue to Magenta (BGR format: Blue=increasing, Green=0, Red=255)
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 * i / BM, 0, 255);
    }
    // MR - Magenta to Red (BGR format: Blue=255-decreasing, Green=0, Red=255)
    for (int i = 0; i < MR; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 - 255 * i / MR, 0, 255);
    }
    
    return colorwheel;
}

cv::Mat RaftPostprocessor::visualizeFlow(const cv::Mat& flow_x, const cv::Mat& flow_y) {
    if (flow_x.empty() || flow_y.empty() || flow_x.size() != flow_y.size()) {
        return cv::Mat();
    }
    
    // Compute magnitude and angle
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_x, flow_y, magnitude, angle);

    // Debug: Check magnitude values before normalization
    double mag_min, mag_max;
    cv::minMaxLoc(magnitude, &mag_min, &mag_max);
    std::cout << "Magnitude range before normalization: " << mag_min << " to " << mag_max << std::endl;
    
    // Normalize magnitude (but clamp to reasonable range first)
    if (mag_max > 0) {
        // Apply magnitude clamp like master branch might do
        magnitude /= mag_max;
    }
    
    // Debug: Check magnitude after normalization
    cv::minMaxLoc(magnitude, &mag_min, &mag_max);
    std::cout << "Magnitude range after normalization: " << mag_min << " to " << mag_max << std::endl;

    // Convert angle to [0, 1] range (matching master branch)
    angle *= (1.0 / (2.0 * CV_PI));
    angle += 0.5; // Shift angle for proper color wheel mapping

    // Apply color wheel
    cv::Mat colorwheel = makeColorwheel();
    const int ncols = colorwheel.rows;
    cv::Mat flow_color(flow_x.size(), CV_8UC3);

    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_x.cols; ++j) {
            float mag = magnitude.at<float>(i, j);
            float ang = angle.at<float>(i, j);

            // Handle angles outside [0,1] range after shift
            while (ang < 0) ang += 1.0f;
            while (ang >= 1.0f) ang -= 1.0f;

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
                float channel_value = (1.0f - f) * col0[ch] + f * col1[ch];
                // Apply magnitude modulation
                if (mag <= 1.0f) {
                    channel_value = 255.0f - mag * (255.0f - channel_value);
                } else {
                    channel_value *= 0.75f;
                }
                color[ch] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, channel_value)));
            }

            flow_color.at<cv::Vec3b>(i, j) = color;
        }
    }

    return flow_color;
}

} // namespace vision_core
