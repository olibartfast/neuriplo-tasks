#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Depth Anything V2 preprocessor
 *
 * Uses RGB conversion, [0,1] normalization and ImageNet normalization.
 */
class DepthAnythingV2Preprocessor : public Preprocessor {
  public:
    explicit DepthAnythingV2Preprocessor(const Size& input_size = Size(518, 518));
};

} // namespace neuriplo_tasks
