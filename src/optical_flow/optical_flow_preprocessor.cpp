#include "vision-core/optical_flow/optical_flow_preprocessor.hpp"

namespace vision_core {

RaftPreprocessor::RaftPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
        input_size,
        ImageFormat::NCHW,
        DataType::FLOAT32,
        true,   // normalize to [0,1]
        false,  // no ImageNet norm (RAFT uses different normalization)
        true    // BGR to RGB
    }) {}

std::vector<std::vector<uint8_t>> RaftPreprocessor::preprocess_pair(
    const cv::Mat& frame1,
    const cv::Mat& frame2) const
{
    std::vector<cv::Mat> frames = {frame1, frame2};
    
    std::vector<std::vector<uint8_t>> preprocessed_frames;
    preprocessed_frames.reserve(2);
    
    for (const auto& frame : frames) {
        cv::Mat processed = frame.clone();
        
        if (config_.bgr_to_rgb && frame.channels() == 3) {
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
        }
        
        cv::resize(processed, processed, config_.input_size, 0, 0, cv::INTER_LINEAR);
        
        // RAFT-specific normalization: (pixel / 255.0 - 0.5) / 0.5
        processed.convertTo(processed, CV_32FC3, 1.0 / 255.0);
        processed = (processed - 0.5) / 0.5;
        
        // Convert to NCHW
        std::vector<cv::Mat> channels;
        cv::split(processed, channels);
        
        std::vector<uint8_t> output;
        for (const auto& channel : channels) {
            const uint8_t* data = channel.data;
            const size_t channel_size = channel.total() * sizeof(float);
            output.insert(output.end(), data, data + channel_size);
        }
        
        preprocessed_frames.push_back(std::move(output));
    }
    
    return preprocessed_frames;
}

} // namespace vision_core
