#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for depth estimation postprocessing
 */
class DepthEstimationPostprocessor {
  public:
    virtual ~DepthEstimationPostprocessor() = default;

    /**
     * @brief Postprocess inference results into depth maps
     * @param depth_output Output tensor data
     * @param shape Output tensor shape
     * @param frame_size Original frame size
     * @return Vector of depth estimation results
     */
    virtual std::vector<DepthEstimation> postprocess(const std::vector<TensorElement>& depth_output,
                                                     const std::vector<int64_t>& shape, const cv::Size& frame_size) = 0;
};

} // namespace neuriplo_tasks
