#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for Gaussian Splatting postprocessors
 *
 * Converts raw network output tensors into a GaussianSplatting result
 * (a flat list of Gaussian3D primitives).
 */
class GaussianSplattingPostprocessor {
  public:
    virtual ~GaussianSplattingPostprocessor() = default;

    /**
     * @brief Postprocess raw tensor data into Gaussian primitives
     * @param tensor_data Flattened tensor values from the network output
     * @param shape        Shape of the output tensor
     * @return GaussianSplatting result containing all predicted Gaussians
     */
    virtual GaussianSplatting postprocess(const std::vector<TensorElement>& tensor_data,
                                          const std::vector<int64_t>& shape) = 0;
};

} // namespace neuriplo_tasks
