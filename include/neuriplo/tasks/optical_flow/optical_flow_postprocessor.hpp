#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for optical flow postprocessing
 */
class OpticalFlowPostprocessor {
  public:
    virtual ~OpticalFlowPostprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param flow_output Output tensor
     * @param shape Output tensor shape
     * @param frame_size Original frame size
     * @return Vector of OpticalFlow results
     */
    virtual std::vector<OpticalFlow> postprocess(const std::vector<TensorElement>& flow_output,
                                                 const std::vector<int64_t>& shape, const Size& frame_size) = 0;
};

} // namespace neuriplo_tasks
