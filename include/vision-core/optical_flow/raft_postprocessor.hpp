#pragma once

#include "vision-core/optical_flow/optical_flow_postprocessor.hpp"
#include <opencv2/core/mat.hpp>

namespace vision_core {

class RaftPostprocessor : public OpticalFlowPostprocessor {
public:
    RaftPostprocessor();

    std::vector<OpticalFlow> postprocess(
        const std::vector<TensorElement>& flow_output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size) override;

private:
    float getTensorFloat(const TensorElement& element);
    cv::Mat visualizeFlow(const cv::Mat& flow_x, const cv::Mat& flow_y);
    cv::Mat makeColorwheel();
};

} // namespace vision_core
