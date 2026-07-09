#include "neuriplo/tasks/core/preprocessor.hpp"

#include "image_ops.hpp"

#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks {

Preprocessor::Preprocessor(const PreprocessConfig& config) : config_(config) {}

void Preprocessor::apply_imagenet_normalization(vision::Image& image) const {
    if (image.channels() != 3 || image.pixelType() != vision::PixelType::Float32) {
        throw std::invalid_argument("Image must be Float32 with 3 channels for ImageNet normalization");
    }
    const int channels = image.channels();
    const std::size_t plane = image.totalPixels();
    float* base = image.data<float>();
    for (int c = 0; c < channels; ++c) {
        const float mean = kImageNetMean[static_cast<std::size_t>(c)];
        const float std_dev = kImageNetStd[static_cast<std::size_t>(c)];
        for (std::size_t i = 0; i < plane; ++i) {
            const std::size_t idx = i * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c);
            base[idx] = (base[idx] - mean) / std_dev;
        }
    }
}

std::vector<uint8_t> Preprocessor::preprocess_image(const vision::ImageView& image, const vision::Size& target_size,
                                                    ImageFormat format, DataType data_type) const {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    vision::Image processed = vision::Image::uninit(image.width(), image.height(), image.channels(), image.pixelType());
    std::memcpy(processed.raw(), image.raw(), image.sizeBytes());

    // Convert BGR to RGB if needed
    if (config_.bgr_to_rgb && processed.channels() == 3) {
        image_ops::swapBgrRgb(processed);
    }

    // Resize
    if (processed.size() != target_size) {
        processed = image_ops::resize(processed, target_size, image_ops::Interpolation::Linear);
    }

    // Normalize to [0, 1] if needed
    if (config_.normalize) {
        processed.convertTo(vision::PixelType::Float32, 1.0 / 255.0);
    } else if (data_type == DataType::FLOAT32) {
        processed.convertTo(vision::PixelType::Float32);
    }

    // Apply ImageNet normalization if requested
    if (config_.apply_imagenet_norm && processed.pixelType() == vision::PixelType::Float32 &&
        processed.channels() == 3) {
        apply_imagenet_normalization(processed);
    }

    // Convert to target format (NCHW planar or NHWC interleaved)
    std::vector<uint8_t> output;
    const std::size_t elem_size = (data_type == DataType::FLOAT32) ? sizeof(float) : sizeof(uint8_t);

    if (format == ImageFormat::NCHW && processed.channels() > 1) {
        // Channel-first: split channels and concatenate
        std::vector<vision::Image> planes = image_ops::splitChannels(processed.view());
        for (const auto& plane : planes) {
            const std::uint8_t* data = plane.raw();
            const std::size_t plane_size = plane.totalPixels() * elem_size;
            output.insert(output.end(), data, data + plane_size);
        }
    } else {
        // Channel-last or single channel: direct copy
        const std::size_t total_size = processed.sizeBytes();
        output.resize(total_size);
        std::memcpy(output.data(), processed.raw(), total_size);
    }

    return output;
}

std::vector<uint8_t> Preprocessor::preprocess(const vision::ImageView& image) const {
    return preprocess_image(image, config_.input_size, config_.format, config_.data_type);
}

std::vector<std::vector<uint8_t>> Preprocessor::preprocess(const std::vector<vision::Image>& images) const {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(images.size());

    for (const auto& image : images) {
        results.push_back(preprocess(image.view()));
    }

    return results;
}

} // namespace neuriplo_tasks
