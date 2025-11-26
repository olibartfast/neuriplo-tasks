#include "vision-core/object_detection/detection_preprocessor.hpp"

namespace vision_core {

// Unified YOLO Preprocessor (handles ALL YOLO variants: v5-v12, v10, NAS)
YoloPreprocessor::YoloPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
        input_size,
        ImageFormat::NCHW,
        DataType::FLOAT32,
        true,   // normalize to [0,1]
        false,  // no ImageNet normalization (YOLO standard)
        true    // BGR to RGB
    }) {}

std::vector<uint8_t> YoloPreprocessor::preprocess(const cv::Mat& image) const {
    // YOLO uses letterbox resizing - pad to maintain aspect ratio
    // This works for ALL YOLO variants (v5-v12, v10, NAS)
    cv::Mat letterbox_image = cv::Mat::zeros(config_.input_size, CV_8UC3);
    
    // Calculate scale
    float scale = std::min(
        static_cast<float>(config_.input_size.width) / image.cols,
        static_cast<float>(config_.input_size.height) / image.rows
    );
    
    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    
    // Center the image
    int x_offset = (config_.input_size.width - new_width) / 2;
    int y_offset = (config_.input_size.height - new_height) / 2;
    
    resized.copyTo(letterbox_image(cv::Rect(x_offset, y_offset, new_width, new_height)));
    
    return preprocess_image(letterbox_image, config_.input_size, config_.format, config_.data_type);
}

// RT-DETR Preprocessor (transformer-based detector)
RtDetrPreprocessor::RtDetrPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
        input_size,
        ImageFormat::NCHW,
        DataType::FLOAT32,
        true,  // normalize
        true,  // ImageNet normalization (transformer models expect this)
        true   // BGR to RGB
    }) {}

// D-FINE Preprocessor (DETR-based detector)
DFinePreprocessor::DFinePreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
        input_size,
        ImageFormat::NCHW,
        DataType::FLOAT32,
        true,  // normalize
        true,  // ImageNet normalization
        true   // BGR to RGB
    }) {}

// RF-DETR Preprocessor (receptive field DETR)
RfDetrPreprocessor::RfDetrPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
        input_size,
        ImageFormat::NCHW,
        DataType::FLOAT32,
        true,  // normalize
        true,  // ImageNet normalization
        true   // BGR to RGB
    }) {}

} // namespace vision_core
