#include "neuriplo/tasks/core/preprocessor.hpp"

#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks {

Preprocessor::Preprocessor(const PreprocessConfig& config) : config_(config) {}

void Preprocessor::apply_imagenet_normalization(cv::Mat& image) const {
    if (image.type() != CV_32FC3) {
        throw std::invalid_argument("Image must be CV_32FC3 for ImageNet normalization");
    }

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    for (size_t i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - kImageNetMean[i]) / kImageNetStd[i];
    }

    cv::merge(channels, image);
}

std::vector<uint8_t> Preprocessor::preprocess_image(const cv::Mat& image, const cv::Size& target_size,
                                                    ImageFormat format, DataType data_type) const {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    cv::Mat processed = image.clone();

    // Convert BGR to RGB if needed
    if (config_.bgr_to_rgb && image.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }

    // Resize
    if (processed.size() != target_size) {
        cv::resize(processed, processed, target_size, 0, 0, cv::INTER_LINEAR);
    }

    // Normalize to [0, 1] if needed
    if (config_.normalize) {
        processed.convertTo(processed, CV_32FC(processed.channels()), 1.0 / 255.0);
    } else if (data_type == DataType::FLOAT32) {
        processed.convertTo(processed, CV_32FC(processed.channels()));
    }

    // Apply ImageNet normalization if requested
    if (config_.apply_imagenet_norm && processed.type() == CV_32FC3) {
        apply_imagenet_normalization(processed);
    }

    // Convert to target format
    std::vector<uint8_t> output;
    const size_t elem_size = (data_type == DataType::FLOAT32) ? sizeof(float) : sizeof(uint8_t);

    if (format == ImageFormat::NCHW && processed.channels() > 1) {
        // Channel-first: split channels and concatenate
        std::vector<cv::Mat> channels;
        cv::split(processed, channels);

        for (const auto& channel : channels) {
            const uint8_t* data = channel.data;
            const size_t channel_size = channel.total() * elem_size;
            output.insert(output.end(), data, data + channel_size);
        }
    } else {
        // Channel-last or single channel: direct copy
        const size_t total_size = processed.total() * processed.elemSize();
        output.resize(total_size);
        std::memcpy(output.data(), processed.data, total_size);
    }

    return output;
}

std::vector<uint8_t> Preprocessor::preprocess(const cv::Mat& image) const {
    return preprocess_image(image, config_.input_size, config_.format, config_.data_type);
}

std::vector<std::vector<uint8_t>> Preprocessor::preprocess(const std::vector<cv::Mat>& images) const {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(images.size());

    for (const auto& image : images) {
        results.push_back(preprocess(image));
    }

    return results;
}

} // namespace neuriplo_tasks
