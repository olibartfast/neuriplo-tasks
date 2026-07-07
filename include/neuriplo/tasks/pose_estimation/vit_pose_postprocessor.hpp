#pragma once

#include "neuriplo/tasks/pose_estimation/pose_postprocessor.hpp"

namespace neuriplo_tasks {

class ViTPosePostprocessor : public PosePostprocessor {
  public:
    ViTPosePostprocessor();

    std::vector<PoseEstimation> postprocess(const std::vector<Tensor>& tensors, const Size& original_size,
                                            const Size& input_size) override;
};

} // namespace neuriplo_tasks
