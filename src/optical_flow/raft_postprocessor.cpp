#include "vision-core/optical_flow/raft_postprocessor.hpp"
#include <stdexcept>

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
    
    const float* data = std::get_if<float>(&flow_output[0]);
    if (!data) return {};
    
    cv::Mat flow_x(height, width, CV_32F);
    cv::Mat flow_y(height, width, CV_32F);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            flow_x.at<float>(y, x) = data[0 * height * width + y * width + x];
            flow_y.at<float>(y, x) = data[1 * height * width + y * width + x];
        }
    }
    
    // Resize to original frame size if needed
    if (frame_size.width != width || frame_size.height != height) {
        cv::resize(flow_x, flow_x, frame_size);
        cv::resize(flow_y, flow_y, frame_size);
        
        // Scale flow values
        flow_x *= static_cast<float>(frame_size.width) / width;
        flow_y *= static_cast<float>(frame_size.height) / height;
    }
    
    OpticalFlow result;
    
    // Merge x and y flows into 2-channel raw_flow
    std::vector<cv::Mat> flow_channels = {flow_x, flow_y};
    cv::merge(flow_channels, result.raw_flow);
    
    // Calculate max displacement
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_x, flow_y, magnitude, angle);
    double max_disp;
    cv::minMaxLoc(magnitude, nullptr, &max_disp);
    result.max_displacement = static_cast<float>(max_disp);
    
    result.flow = visualizeFlow(flow_x, flow_y);
    
    return {result};
}

float RaftPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

cv::Mat RaftPostprocessor::makeColorwheel() {
    // Constants for color wheel (based on Middlebury flow color coding)
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    // RY - Red to Yellow
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255 * i / RY, 255);
    }
    // YG - Yellow to Green
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255, 255 - 255 * i / YG);
    }
    // GC - Green to Cyan
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 * i / GC, 255, 0);
    }
    // CB - Cyan to Blue
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 255 - 255 * i / CB, 0);
    }
    // BM - Blue to Magenta
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 0, 255 * i / BM);
    }
    // MR - Magenta to Red
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

    // Normalize magnitude
    double mag_max;
    cv::minMaxLoc(magnitude, nullptr, &mag_max);
    if (mag_max > 0) {
        magnitude /= mag_max;
    }

    // Convert angle to [0, 1] range
    angle *= (1.0 / (2.0 * CV_PI));

    // Apply color wheel
    cv::Mat colorwheel = makeColorwheel();
    const int ncols = colorwheel.rows;
    cv::Mat flow_color(flow_x.size(), CV_8UC3);

    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_x.cols; ++j) {
            float mag = magnitude.at<float>(i, j);
            float ang = angle.at<float>(i, j);

            // Handle negative angles
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
