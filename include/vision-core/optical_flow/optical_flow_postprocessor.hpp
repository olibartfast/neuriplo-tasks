#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"

#include <opencv2/core/types.hpp>
#include <vector>

namespace vision_core {

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
                                                 const std::vector<int64_t>& shape, const cv::Size& frame_size) = 0;
};

} // namespace vision_core
