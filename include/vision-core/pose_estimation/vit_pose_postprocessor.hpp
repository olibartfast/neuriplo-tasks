#pragma once

#include "vision-core/pose_estimation/pose_postprocessor.hpp"

namespace vision_core {

class ViTPosePostprocessor : public PosePostprocessor {
public:
    ViTPosePostprocessor();

    std::vector<PoseEstimation> postprocess(
        const std::vector<Tensor>& tensors,
        const cv::Size& original_size,
        const cv::Size& input_size) override;
};

} // namespace vision_core
