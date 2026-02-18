#include "vision-core/optical_flow/raft_postprocessor.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace vision_core {

RaftPostprocessor::RaftPostprocessor() {}

std::vector<OpticalFlow> RaftPostprocessor::postprocess(const std::vector<TensorElement>& flow_output,
                                                        const std::vector<int64_t>& shape, const cv::Size& frame_size) {

    if (flow_output.empty() || shape.empty()) {
        return {};
    }

    // RAFT output: [1, 2, H, W] (dx, dy)
    if (shape.size() < 4)
        return {};

    int channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);

    if (channels != 2)
        return {};

    // Robust tensor data access - handle float tensors
    const float* data = std::get_if<float>(&flow_output[0]);
    if (!data) {
        // Fallback: convert other types to float
        static std::vector<float> temp_buffer;
        temp_buffer.resize(flow_output.size());
        for (size_t i = 0; i < flow_output.size(); ++i) {
            temp_buffer[i] = getTensorFloat(flow_output[i]);
        }
        data = temp_buffer.data();
    }

    // Create flow matrix [H, W, 2] like master branch
    cv::Mat flow(height, width, CV_32FC2);
    float* flow_ptr = reinterpret_cast<float*>(flow.data);

    // Channel offset logic matching master branch
    const int u_channel_offset = 0;
    const int v_channel_offset = height * width;

    // Reconstruct flow matrix using direct TensorElement access (like master branch)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            flow_ptr[y * width * 2 + x * 2] =
                getTensorFloat(flow_output[static_cast<size_t>(u_channel_offset + y * width + x)]);
            flow_ptr[y * width * 2 + x * 2 + 1] =
                getTensorFloat(flow_output[static_cast<size_t>(v_channel_offset + y * width + x)]);
        }
    }

    // Resize to original frame size if needed (with proper flow scaling)
    if (frame_size.width != width || frame_size.height != height) {
        cv::Mat resized_flow;
        cv::resize(flow, resized_flow, frame_size);

        // Scale flow values proportionally
        std::vector<cv::Mat> flow_channels;
        cv::split(resized_flow, flow_channels);
        flow_channels[0] *= static_cast<float>(frame_size.width) / static_cast<float>(width);   // Scale U
        flow_channels[1] *= static_cast<float>(frame_size.height) / static_cast<float>(height); // Scale V
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
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

cv::Mat RaftPostprocessor::makeColorwheel() {
    // Constants for color wheel (matching master branch exactly)
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    // RY - Red to Yellow (BGR format: Blue=255, Green=increasing, Red=0)
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, static_cast<uchar>(255 * i / RY), 0);
    }
    // YG - Yellow to Green (BGR format: Blue=255-decreasing, Green=255, Red=0)
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(static_cast<uchar>(255 - 255 * i / YG), 255, 0);
    }
    // GC - Green to Cyan (BGR format: Blue=0, Green=255, Red=increasing)
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255, static_cast<uchar>(255 * i / GC));
    }
    // CB - Cyan to Blue (BGR format: Blue=0, Green=255-decreasing, Red=255)
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, static_cast<uchar>(255 - 255 * i / CB), 255);
    }
    // BM - Blue to Magenta (BGR format: Blue=increasing, Green=0, Red=255)
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(static_cast<uchar>(255 * i / BM), 0, 255);
    }
    // MR - Magenta to Red (BGR format: Blue=255-decreasing, Green=0, Red=255)
    for (int i = 0; i < MR; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(static_cast<uchar>(255 - 255 * i / MR), 0, 255);
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
            while (ang < 0)
                ang += 1.0f;
            while (ang >= 1.0f)
                ang -= 1.0f;

            // Find nearest colors in the wheel
            int k0 = static_cast<int>(ang * static_cast<float>(ncols - 1));
            int k1 = (k0 + 1) % ncols;
            float f = (ang * static_cast<float>(ncols - 1)) - static_cast<float>(k0);

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
