#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision_core {

/**
 * @brief Interface for object detection postprocessing
 */
class Postprocessor {
public:
    virtual ~Postprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param infer_results Inference output tensors
     * @param infer_shapes Output tensor shapes
     * @param frame_size Original frame size
     * @return Vector of Detection results
     */
    virtual std::vector<Detection> postprocess(
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes,
        const cv::Size& frame_size) = 0;
};

} // namespace vision_core
