#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"

#include <vector>

namespace vision_core {

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

} // namespace vision_core
