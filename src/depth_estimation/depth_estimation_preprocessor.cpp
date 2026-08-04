#include "neuriplo/tasks/depth_estimation/depth_estimation_preprocessor.hpp"

#include "image_ops.hpp"

#include <algorithm>
#include <cmath>

namespace neuriplo_tasks {

DepthAnythingV2Preprocessor::DepthAnythingV2Preprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize to [0,1]
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

YoloDepthPreprocessor::YoloDepthPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{input_size, ImageFormat::NCHW, DataType::FLOAT32, true, false, true}) {}

std::vector<uint8_t> YoloDepthPreprocessor::preprocess(const vision::ImageView& image) const {
    if (image.empty()) {
        return Preprocessor::preprocess(image);
    }

    vision::Image letterbox =
        vision::Image::uninit(config_.input_size.width, config_.input_size.height, 3, vision::PixelType::UInt8);
    std::fill(letterbox.raw(), letterbox.raw() + letterbox.sizeBytes(), static_cast<uint8_t>(114));

    const float scale = std::min(static_cast<float>(config_.input_size.width) / static_cast<float>(image.width()),
                                 static_cast<float>(config_.input_size.height) / static_cast<float>(image.height()));
    const int resized_width = static_cast<int>(std::round(static_cast<float>(image.width()) * scale));
    const int resized_height = static_cast<int>(std::round(static_cast<float>(image.height()) * scale));
    vision::Image resized = image_ops::resize(image, resized_width, resized_height, image_ops::Interpolation::Linear);

    const int x_offset = (config_.input_size.width - resized_width) / 2;
    const int y_offset = (config_.input_size.height - resized_height) / 2;
    image_ops::copyRegion(resized.view(), vision::Rect(0, 0, resized_width, resized_height), letterbox,
                          vision::Rect(x_offset, y_offset, resized_width, resized_height));

    return preprocess_image(letterbox.view(), config_.input_size, config_.format, config_.data_type);
}

} // namespace neuriplo_tasks
