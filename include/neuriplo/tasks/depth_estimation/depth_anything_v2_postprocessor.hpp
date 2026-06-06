#pragma once

#include "neuriplo/tasks/depth_estimation/depth_estimation_postprocessor.hpp"

namespace neuriplo_tasks {

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
};

} // namespace neuriplo_tasks
