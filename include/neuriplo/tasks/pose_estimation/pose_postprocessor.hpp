#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for pose estimation postprocessing
 */
class PosePostprocessor {
  public:
    virtual ~PosePostprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param tensors Inference output tensors
     * @param original_size Original image size
     * @param input_size Network input size
     * @return Vector of PoseEstimation results
     */
    virtual std::vector<PoseEstimation> postprocess(const std::vector<Tensor>& tensors, const cv::Size& original_size,
                                                    const cv::Size& input_size) = 0;
};

} // namespace neuriplo_tasks
