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
    
    result.flow = visualizeFlow(flow_x, flow_y);
    
    return {result};
}

float RaftPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

cv::Mat RaftPostprocessor::visualizeFlow(const cv::Mat& flow_x, const cv::Mat& flow_y) {
    // TODO: Implement flow visualization
    return cv::Mat();
}

} // namespace vision_core
