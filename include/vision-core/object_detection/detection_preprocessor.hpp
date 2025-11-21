#pragma once

#include "vision-core/core/preprocessor.hpp"

namespace vision_core {

/**
 * @brief YOLO-specific preprocessor
 * 
 * Handles letterbox resizing and normalization for YOLO models (v5-v12).
 */
class YoloPreprocessor : public Preprocessor {
public:
    explicit YoloPreprocessor(const cv::Size& input_size);

    [[nodiscard]] std::vector<uint8_t> preprocess(const cv::Mat& image) const override;
};

/**
 * @brief YOLOv10-specific preprocessor
 */
class Yolov10Preprocessor : public Preprocessor {
public:
    explicit Yolov10Preprocessor(const cv::Size& input_size);
};

/**
 * @brief RT-DETR preprocessor
 */
class RtDetrPreprocessor : public Preprocessor {
public:
    explicit RtDetrPreprocessor(const cv::Size& input_size);
};

/**
 * @brief YOLO-NAS preprocessor
 */
class YoloNasPreprocessor : public Preprocessor {
public:
    explicit YoloNasPreprocessor(const cv::Size& input_size);
};

/**
 * @brief D-FINE preprocessor
 */
class DFinePreprocessor : public Preprocessor {
public:
    explicit DFinePreprocessor(const cv::Size& input_size);
};

/**
 * @brief RF-DETR preprocessor
 */
class RfDetrPreprocessor : public Preprocessor {
public:
    explicit RfDetrPreprocessor(const cv::Size& input_size);
};

} // namespace vision_core
