#include "vision-core/optical_flow/optical_flow_preprocessor.hpp"

namespace vision_core {

RaftPreprocessor::RaftPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          false, // NO base normalization (RAFT does custom [-1,1] normalization)
          false, // no ImageNet norm
          true   // BGR to RGB
      }) {}

std::vector<std::vector<uint8_t>> RaftPreprocessor::preprocess_pair(const cv::Mat& frame1,
                                                                    const cv::Mat& frame2) const {
    std::vector<cv::Mat> frames = {frame1, frame2};

    std::vector<std::vector<uint8_t>> preprocessed_frames;
    preprocessed_frames.reserve(2);

    for (const auto& frame : frames) {
        cv::Mat processed;

        // Color space conversion (BGR to RGB)
        if (config_.bgr_to_rgb && frame.channels() == 3) {
            cv::cvtColor(frame, processed, cv::COLOR_BGR2RGB);
        } else {
            processed = frame.clone();
        }

        // Resize with appropriate interpolation
        int interpolation = cv::INTER_LINEAR;
        if (frame.size().width > config_.input_size.width || frame.size().height > config_.input_size.height) {
            interpolation = cv::INTER_AREA; // Better for downsampling
        }
        cv::resize(processed, processed, config_.input_size, 0, 0, interpolation);

        // Normalization: (pixel / 255.0 - 0.5) / 0.5 -> [-1, 1] range
        processed.convertTo(processed, CV_32FC3, 1.0 / 255.0);
        processed = (processed - 0.5) / 0.5;

        // Convert to NCHW format if required
        std::vector<uint8_t> output;
        if (config_.format == ImageFormat::NCHW) {
            std::vector<cv::Mat> channels;
            cv::split(processed, channels);

            for (const auto& channel : channels) {
                const float* data = reinterpret_cast<const float*>(channel.data);
                const size_t channel_size = channel.total() * sizeof(float);
                const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
                output.insert(output.end(), byte_data, byte_data + channel_size);
            }
        } else {
            // HWC format
            const float* data = reinterpret_cast<const float*>(processed.data);
            const size_t total_size = processed.total() * static_cast<size_t>(processed.channels()) * sizeof(float);
            const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
            output.insert(output.end(), byte_data, byte_data + total_size);
        }

        preprocessed_frames.push_back(std::move(output));
    }

    return preprocessed_frames;
}

} // namespace vision_core
