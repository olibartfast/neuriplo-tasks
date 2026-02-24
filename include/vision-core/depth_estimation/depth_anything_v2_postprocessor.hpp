#pragma once

#include "vision-core/depth_estimation/depth_estimation_postprocessor.hpp"

namespace vision_core {

/**
 * @brief Depth Anything V2 postprocessor
 *
 * Supported output layouts:
 * - [B, H, W]
 * - [B, 1, H, W]
 * - [H, W]
 */
class DepthAnythingV2Postprocessor : public DepthEstimationPostprocessor {
  public:
    std::vector<DepthEstimation> postprocess(const std::vector<TensorElement>& depth_output,
                                             const std::vector<int64_t>& shape, const cv::Size& frame_size) override;

  private:
    static float getTensorFloat(const TensorElement& element);
};

} // namespace vision_core
