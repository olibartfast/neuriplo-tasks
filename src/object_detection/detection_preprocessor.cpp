#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"

#include "image_ops.hpp"

namespace neuriplo_tasks {

// Unified YOLO Preprocessor (handles ALL YOLO variants: v5-v12, v10, NAS)
YoloPreprocessor::YoloPreprocessor(const Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true,  // normalize to [0,1]
          false, // no ImageNet normalization (YOLO standard)
          true   // BGR to RGB
      }) {}

std::vector<uint8_t> YoloPreprocessor::preprocess(const ImageView& image) const {
    Image letterbox_image = Image::zeros(config_.input_size.width, config_.input_size.height, 3, PixelType::UInt8);

    float scale = std::min(static_cast<float>(config_.input_size.width) / static_cast<float>(image.width()),
                           static_cast<float>(config_.input_size.height) / static_cast<float>(image.height()));

    int new_width = static_cast<int>(static_cast<float>(image.width()) * scale);
    int new_height = static_cast<int>(static_cast<float>(image.height()) * scale);

    Image resized = image_ops::resize(image, new_width, new_height, image_ops::Interpolation::Linear);

    int x_offset = (config_.input_size.width - new_width) / 2;
    int y_offset = (config_.input_size.height - new_height) / 2;

    image_ops::copyRegion(resized.view(), BoundingBox(0, 0, new_width, new_height), letterbox_image,
                          BoundingBox(x_offset, y_offset, new_width, new_height));

    return preprocess_image(letterbox_image.view(), config_.input_size, config_.format, config_.data_type);
}

// RT-DETR Preprocessor (transformer-based detector)
RtDetrPreprocessor::RtDetrPreprocessor(const Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize
          true, // ImageNet normalization (transformer models expect this)
          true  // BGR to RGB
      }) {}

// D-FINE Preprocessor (DETR-based detector)
DFinePreprocessor::DFinePreprocessor(const Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

// RF-DETR Preprocessor (receptive field DETR)
RfDetrPreprocessor::RfDetrPreprocessor(const Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

} // namespace neuriplo_tasks
