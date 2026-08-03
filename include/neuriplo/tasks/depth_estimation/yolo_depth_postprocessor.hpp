#pragma once

#include "neuriplo/tasks/depth_estimation/depth_estimation_postprocessor.hpp"

namespace neuriplo_tasks {

class YoloDepthPostprocessor : public DepthEstimationPostprocessor {
  public:
    std::vector<DepthEstimation> postprocess(const std::vector<TensorElement>& depth_output,
                                             const std::vector<int64_t>& shape, const Size& frame_size) override;
};

} // namespace neuriplo_tasks
