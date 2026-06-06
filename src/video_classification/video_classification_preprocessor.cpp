#include "neuriplo/tasks/video_classification/video_classification_preprocessor.hpp"

#include <cstring>

namespace neuriplo_tasks {

// ============ VideoMAEPreprocessor ============

VideoMAEPreprocessor::VideoMAEPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          false, // We handle normalization manually
          false,
          true // BGR to RGB
      }) {}

std::vector<uint8_t> VideoMAEPreprocessor::preprocess(const cv::Mat& image) const {
    cv::Mat processed;

    // BGR to RGB
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);

    // Direct resize to target size
    cv::resize(processed, processed, config_.input_size, 0, 0, cv::INTER_LINEAR);

    // Convert to float and rescale /255
    processed.convertTo(processed, CV_32FC3, 1.0 / 255.0);

    // Apply normalization: (pixel - mean) / std
    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    for (size_t i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - kMean[i]) / kStd[i];
    }
    cv::merge(channels, processed);

    // Convert to NCHW format
    std::vector<uint8_t> output;
    cv::split(processed, channels);
    for (const auto& channel : channels) {
        const auto* data = reinterpret_cast<const uint8_t*>(channel.data);
        size_t channel_size = channel.total() * sizeof(float);
        output.insert(output.end(), data, data + channel_size);
    }

    return output;
}

// ============ VivitPreprocessor ============

VivitPreprocessor::VivitPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{input_size, ImageFormat::NCHW, DataType::FLOAT32, false, false, true}) {}

std::vector<uint8_t> VivitPreprocessor::preprocess(const cv::Mat& image) const {
    cv::Mat processed;

    // BGR to RGB
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);

    // Aspect-preserving resize: shortest edge to 256
    int h = processed.rows;
    int w = processed.cols;
    float scale = static_cast<float>(kShortestEdge) / static_cast<float>(std::min(h, w));
    int new_h = static_cast<int>(static_cast<float>(h) * scale);
    int new_w = static_cast<int>(static_cast<float>(w) * scale);
    cv::resize(processed, processed, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Center crop to target size
    int crop_x = (processed.cols - config_.input_size.width) / 2;
    int crop_y = (processed.rows - config_.input_size.height) / 2;
    processed = processed(cv::Rect(crop_x, crop_y, config_.input_size.width, config_.input_size.height)).clone();

    // Convert to float and rescale: pixel * (1/127.5) - 1
    processed.convertTo(processed, CV_32FC3, 1.0 / 127.5, -1.0);

    // Apply ImageNet normalization: (pixel - mean) / std
    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    for (size_t i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - kMean[i]) / kStd[i];
    }
    cv::merge(channels, processed);

    // Convert to NCHW format
    std::vector<uint8_t> output;
    cv::split(processed, channels);
    for (const auto& channel : channels) {
        const auto* data = reinterpret_cast<const uint8_t*>(channel.data);
        size_t channel_size = channel.total() * sizeof(float);
        output.insert(output.end(), data, data + channel_size);
    }

    return output;
}

// ============ TimeSformerPreprocessor ============

TimeSformerPreprocessor::TimeSformerPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{input_size, ImageFormat::NCHW, DataType::FLOAT32, false, false, true}) {}

std::vector<uint8_t> TimeSformerPreprocessor::preprocess(const cv::Mat& image) const {
    cv::Mat processed;

    // BGR to RGB
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);

    // Aspect-preserving resize: shortest edge to 224
    int h = processed.rows;
    int w = processed.cols;
    float scale = static_cast<float>(kShortestEdge) / static_cast<float>(std::min(h, w));
    int new_h = static_cast<int>(static_cast<float>(h) * scale);
    int new_w = static_cast<int>(static_cast<float>(w) * scale);
    cv::resize(processed, processed, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Center crop to target size
    int crop_x = (processed.cols - config_.input_size.width) / 2;
    int crop_y = (processed.rows - config_.input_size.height) / 2;
    processed = processed(cv::Rect(crop_x, crop_y, config_.input_size.width, config_.input_size.height)).clone();

    // Convert to float and rescale /255
    processed.convertTo(processed, CV_32FC3, 1.0 / 255.0);

    // Apply custom normalization: (pixel - mean) / std
    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    for (size_t i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - kMean[i]) / kStd[i];
    }
    cv::merge(channels, processed);

    // Convert to NCHW format
    std::vector<uint8_t> output;
    cv::split(processed, channels);
    for (const auto& channel : channels) {
        const auto* data = reinterpret_cast<const uint8_t*>(channel.data);
        size_t channel_size = channel.total() * sizeof(float);
        output.insert(output.end(), data, data + channel_size);
    }

    return output;
}

} // namespace neuriplo_tasks
