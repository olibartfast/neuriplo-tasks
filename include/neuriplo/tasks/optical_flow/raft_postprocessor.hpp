#pragma once

#include "neuriplo/tasks/optical_flow/optical_flow_postprocessor.hpp"

#include <opencv2/core/mat.hpp>

namespace neuriplo_tasks {

class RaftPostprocessor : public OpticalFlowPostprocessor {
  public:
    RaftPostprocessor();

    std::vector<OpticalFlow> postprocess(const std::vector<TensorElement>& flow_output,
                                         const std::vector<int64_t>& shape, const cv::Size& frame_size) override;

  private:
    cv::Mat visualizeFlow(const cv::Mat& flow_x, const cv::Mat& flow_y);
    cv::Mat makeColorwheel();
};

} // namespace neuriplo_tasks
