#pragma once

#include "vision-core/core/preprocessor.hpp"

namespace vision_core {

/**
 * @brief Generic classifier preprocessor
 */
class ClassifierPreprocessor : public Preprocessor {
public:
    explicit ClassifierPreprocessor(const cv::Size& input_size, bool use_imagenet_norm = false);
};

/**
 * @brief Torchvision models preprocessor
 * Uses ImageNet normalization
 */
class TorchvisionPreprocessor : public Preprocessor {
public:
    explicit TorchvisionPreprocessor(const cv::Size& input_size = cv::Size(224, 224));
};

/**
 * @brief TensorFlow/Keras models preprocessor
 * No ImageNet normalization, just [0, 1] scaling
 */
class TensorflowPreprocessor : public Preprocessor {
public:
    explicit TensorflowPreprocessor(const cv::Size& input_size = cv::Size(224, 224));
};

/**
 * @brief ViT (Vision Transformer) preprocessor
 * Uses ImageNet normalization
 */
class ViTPreprocessor : public Preprocessor {
public:
    explicit ViTPreprocessor(const cv::Size& input_size = cv::Size(224, 224));
};

} // namespace vision_core
