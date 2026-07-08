#include "neuriplo/tasks/video_classification/video_classification_preprocessor.hpp"

#include "image_ops.hpp"

#include <algorithm>
#include <cstring>

namespace neuriplo_tasks {

namespace {

void applyChannelNormalization(vision::Image& processed, const float mean[3], const float std_dev[3]) {
    std::vector<vision::Image> channels = image_ops::splitChannels(processed.view());
    for (size_t ci = 0; ci < channels.size() && ci < 3; ++ci) {
        float* p = channels[ci].data<float>();
        const size_t count = channels[ci].totalPixels();
        for (size_t px = 0; px < count; ++px) {
            p[px] = (p[px] - mean[ci]) / std_dev[ci];
        }
    }
    processed = image_ops::mergeChannels(channels);
}

std::vector<uint8_t> splitToNCHW(const vision::Image& processed) {
    std::vector<uint8_t> output;
    std::vector<vision::Image> channels = image_ops::splitChannels(processed.view());
    for (const auto& channel : channels) {
        const auto* data = reinterpret_cast<const uint8_t*>(channel.data<float>());
        size_t channel_size = channel.totalPixels() * sizeof(float);
        output.insert(output.end(), data, data + channel_size);
    }
    return output;
}

} // namespace

// ============ VideoMAEPreprocessor ============

VideoMAEPreprocessor::VideoMAEPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          false, // We handle normalization manually
          false,
          true // BGR to RGB
      }) {}

std::vector<uint8_t> VideoMAEPreprocessor::preprocess(const vision::ImageView& image) const {
    vision::Image processed = vision::Image(image.width(), image.height(), image.channels(), image.pixelType());
    std::memcpy(processed.raw(), image.raw(), image.sizeBytes());

    if (config_.bgr_to_rgb && processed.channels() == 3) {
        image_ops::swapBgrRgb(processed);
    }

    processed = image_ops::resize(processed, config_.input_size, image_ops::Interpolation::Linear);

    processed.convertTo(vision::PixelType::Float32, 1.0 / 255.0);

    applyChannelNormalization(processed, kMean.data(), kStd.data());

    return splitToNCHW(processed);
}

// ============ VivitPreprocessor ============

VivitPreprocessor::VivitPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{input_size, ImageFormat::NCHW, DataType::FLOAT32, false, false, true}) {}

std::vector<uint8_t> VivitPreprocessor::preprocess(const vision::ImageView& image) const {
    vision::Image processed = vision::Image(image.width(), image.height(), image.channels(), image.pixelType());
    std::memcpy(processed.raw(), image.raw(), image.sizeBytes());

    if (config_.bgr_to_rgb && processed.channels() == 3) {
        image_ops::swapBgrRgb(processed);
    }

    int h = processed.height();
    int w = processed.width();
    float scale = static_cast<float>(kShortestEdge) / static_cast<float>(std::min(h, w));
    int new_h = static_cast<int>(static_cast<float>(h) * scale);
    int new_w = static_cast<int>(static_cast<float>(w) * scale);
    processed = image_ops::resize(processed, new_w, new_h, image_ops::Interpolation::Linear);

    int crop_x = (processed.width() - config_.input_size.width) / 2;
    int crop_y = (processed.height() - config_.input_size.height) / 2;

    vision::Image cropped = vision::Image::uninit(config_.input_size.width, config_.input_size.height,
                                                  processed.channels(), processed.pixelType());
    image_ops::copyRegion(processed.view(),
                          vision::Rect(crop_x, crop_y, config_.input_size.width, config_.input_size.height), cropped,
                          vision::Rect(0, 0, config_.input_size.width, config_.input_size.height));
    processed = std::move(cropped);

    processed.convertTo(vision::PixelType::Float32, 1.0 / 127.5, -1.0);

    applyChannelNormalization(processed, kMean.data(), kStd.data());

    return splitToNCHW(processed);
}

// ============ TimeSformerPreprocessor ============

TimeSformerPreprocessor::TimeSformerPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{input_size, ImageFormat::NCHW, DataType::FLOAT32, false, false, true}) {}

std::vector<uint8_t> TimeSformerPreprocessor::preprocess(const vision::ImageView& image) const {
    vision::Image processed = vision::Image(image.width(), image.height(), image.channels(), image.pixelType());
    std::memcpy(processed.raw(), image.raw(), image.sizeBytes());

    if (config_.bgr_to_rgb && processed.channels() == 3) {
        image_ops::swapBgrRgb(processed);
    }

    int h = processed.height();
    int w = processed.width();
    float scale = static_cast<float>(kShortestEdge) / static_cast<float>(std::min(h, w));
    int new_h = static_cast<int>(static_cast<float>(h) * scale);
    int new_w = static_cast<int>(static_cast<float>(w) * scale);
    processed = image_ops::resize(processed, new_w, new_h, image_ops::Interpolation::Linear);

    int crop_x = (processed.width() - config_.input_size.width) / 2;
    int crop_y = (processed.height() - config_.input_size.height) / 2;

    vision::Image cropped = vision::Image::uninit(config_.input_size.width, config_.input_size.height,
                                                  processed.channels(), processed.pixelType());
    image_ops::copyRegion(processed.view(),
                          vision::Rect(crop_x, crop_y, config_.input_size.width, config_.input_size.height), cropped,
                          vision::Rect(0, 0, config_.input_size.width, config_.input_size.height));
    processed = std::move(cropped);

    processed.convertTo(vision::PixelType::Float32, 1.0 / 255.0);

    applyChannelNormalization(processed, kMean.data(), kStd.data());

    return splitToNCHW(processed);
}

} // namespace neuriplo_tasks
