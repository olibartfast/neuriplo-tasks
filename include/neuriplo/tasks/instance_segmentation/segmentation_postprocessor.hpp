#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <opencv2/core/types.hpp>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for instance segmentation postprocessing
 */
class SegmentationPostprocessor {
  public:
    virtual ~SegmentationPostprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param tensors Vector of output tensors with data and shapes
     * @param frame_size Original frame size
     * @return Vector of InstanceSegmentation results
     */
    virtual std::vector<InstanceSegmentation> postprocess(const std::vector<Tensor>& tensors,
                                                          const cv::Size& frame_size) = 0;
};

} // namespace neuriplo_tasks
