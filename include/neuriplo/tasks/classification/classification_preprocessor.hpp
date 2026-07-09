#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Generic classifier preprocessor
 */
class ClassifierPreprocessor : public Preprocessor {
  public:
    explicit ClassifierPreprocessor(const Size& input_size, bool use_imagenet_norm = false);
};

/**
 * @brief Torchvision models preprocessor
 * Uses ImageNet normalization
 */
class TorchvisionPreprocessor : public Preprocessor {
  public:
    explicit TorchvisionPreprocessor(const Size& input_size = Size(224, 224));
};

/**
 * @brief TensorFlow/Keras models preprocessor
 * No ImageNet normalization, just [0, 1] scaling
 */
class TensorflowPreprocessor : public Preprocessor {
  public:
    explicit TensorflowPreprocessor(const Size& input_size = Size(224, 224));
};

/**
 * @brief ViT (Vision Transformer) preprocessor
 * Uses ImageNet normalization
 */
class ViTPreprocessor : public Preprocessor {
  public:
    explicit ViTPreprocessor(const Size& input_size = Size(224, 224));
};

} // namespace neuriplo_tasks
