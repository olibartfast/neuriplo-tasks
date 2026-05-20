#pragma once

#include "vision-core/core/preprocessor.hpp"

namespace vision_core {

/**
 * @brief Preprocessor for LGM / GRM Gaussian Splatting models
 *
 * Expects a single input image that is:
 * - Resized to the model's input resolution (default 256x256)
 * - Converted BGR→RGB
 * - Normalised to [0, 1] with ImageNet mean/std
 * - Packed in NCHW layout as float32
 *
 * Multi-view models (e.g. the 4-view LGM path) call preprocess() once
 * per view; concatenation across views is handled by the inference engine.
 */
class GaussianSplattingPreprocessor : public Preprocessor {
  public:
    explicit GaussianSplattingPreprocessor(const cv::Size& input_size = cv::Size(256, 256));
};

} // namespace vision_core
