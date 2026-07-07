#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <vector>

namespace neuriplo_tasks {

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
    virtual std::vector<Detection> postprocess(const std::vector<Tensor>& tensors, const Size& frame_size) = 0;
};

} // namespace neuriplo_tasks
