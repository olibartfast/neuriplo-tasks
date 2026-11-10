#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Preprocessor for appearance-based gaze estimation (ETH-XGaze style).
 *
 * Expects a normalized face or eye crop that is:
 * - Resized to the model's input resolution (default 224x224)
 * - Converted BGR to RGB
 * - Normalised to [0, 1] with ImageNet mean/std
 * - Packed in NCHW layout as float32
 */
class GazePreprocessor : public Preprocessor {
  public:
    explicit GazePreprocessor(const Size& input_size = Size(224, 224));
};

} // namespace neuriplo_tasks
