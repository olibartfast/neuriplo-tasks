#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision_core {

/**
 * @brief Interface for object detection postprocessing
 */
class Postprocessor {
  public:
    virtual ~Postprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param tensors Inference output tensors with data and shapes
     * @param frame_size Original frame size
     * @return Vector of Detection results
     */
    virtual std::vector<Detection> postprocess(const std::vector<Tensor>& tensors, const cv::Size& frame_size) = 0;
};

} // namespace vision_core
