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
 * Raw BGR pixels in [0, 255], with layout and dtype selected by the task metadata
 */
class TensorflowPreprocessor : public Preprocessor {
  public:
    explicit TensorflowPreprocessor(const Size& input_size = Size(224, 224));
    TensorflowPreprocessor(const Size& input_size, ImageFormat format, DataType data_type);
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
