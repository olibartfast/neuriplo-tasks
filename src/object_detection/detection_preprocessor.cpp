#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"

#include "image_ops.hpp"

namespace neuriplo_tasks {

// Unified YOLO Preprocessor (handles ALL YOLO variants: v5-v12, v10, NAS)
YoloPreprocessor::YoloPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true,  // normalize to [0,1]
          false, // no ImageNet normalization (YOLO standard)
          true   // BGR to RGB
      }) {}

std::vector<uint8_t> YoloPreprocessor::preprocess(const vision::ImageView& image) const {
    vision::Image letterbox_image =
        vision::Image::zeros(config_.input_size.width, config_.input_size.height, 3, vision::PixelType::UInt8);

    float scale = std::min(static_cast<float>(config_.input_size.width) / static_cast<float>(image.width()),
                           static_cast<float>(config_.input_size.height) / static_cast<float>(image.height()));

    int new_width = static_cast<int>(static_cast<float>(image.width()) * scale);
    int new_height = static_cast<int>(static_cast<float>(image.height()) * scale);

    vision::Image resized = image_ops::resize(image, new_width, new_height, image_ops::Interpolation::Linear);

    int x_offset = (config_.input_size.width - new_width) / 2;
    int y_offset = (config_.input_size.height - new_height) / 2;

    image_ops::copyRegion(resized.view(), vision::Rect(0, 0, new_width, new_height), letterbox_image,
                          vision::Rect(x_offset, y_offset, new_width, new_height));

    return preprocess_image(letterbox_image.view(), config_.input_size, config_.format, config_.data_type);
}

// RT-DETR Preprocessor (transformer-based detector)
// RT-DETR, RT-DETRv2, D-FINE and DEIM all scale to [0,1] and stop there; none of
// them subtract ImageNet statistics. Applying them anyway leaves the detector
// running on out-of-distribution input: it still fires, but on a crowded street
// it drops from a full set of boxes to a handful of badly placed ones.
RtDetrPreprocessor::RtDetrPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true,  // normalize to [0,1]
          false, // no ImageNet normalization
          true   // BGR to RGB
      }) {}

// D-FINE Preprocessor (DETR-based detector). Same convention as RT-DETR above.
DFinePreprocessor::DFinePreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true,  // normalize to [0,1]
          false, // no ImageNet normalization
          true   // BGR to RGB
      }) {}

// EdgeCrafter Preprocessor — retains ImageNet normalization, see the header.
EdgeCrafterPreprocessor::EdgeCrafterPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

// RF-DETR Preprocessor (receptive field DETR)
RfDetrPreprocessor::RfDetrPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

} // namespace neuriplo_tasks
