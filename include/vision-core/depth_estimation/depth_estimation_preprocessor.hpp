#pragma once

#include "vision-core/core/preprocessor.hpp"

namespace vision_core {

/**
 * @brief Depth Anything V2 preprocessor
 *
 * Uses RGB conversion, [0,1] normalization and ImageNet normalization.
 */
class DepthAnythingV2Preprocessor : public Preprocessor {
  public:
    explicit DepthAnythingV2Preprocessor(const cv::Size& input_size = cv::Size(518, 518));
};

} // namespace vision_core
