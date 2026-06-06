#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace neuriplo_tasks {

class OpenVocabPostprocessor {
  public:
    virtual ~OpenVocabPostprocessor() = default;

    virtual std::vector<OpenVocabDetection> postprocess(const std::vector<Tensor>& tensors,
                                                        const cv::Size& frame_size) = 0;
};

} // namespace neuriplo_tasks
