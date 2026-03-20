#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision_core {

class OpenVocabPostprocessor {
  public:
    virtual ~OpenVocabPostprocessor() = default;

    virtual std::vector<OpenVocabDetection> postprocess(const std::vector<Tensor>& tensors,
                                                        const cv::Size& frame_size) = 0;
};

} // namespace vision_core
