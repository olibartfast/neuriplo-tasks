#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"
#include <vector>
#include <opencv2/core/types.hpp>

namespace vision_core {

/**
 * @brief Interface for instance segmentation postprocessing
 */
class SegmentationPostprocessor {
public:
    virtual ~SegmentationPostprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param infer_results Vector of output tensors
     * @param infer_shapes Vector of output tensor shapes
     * @param frame_size Original frame size
     * @return Vector of InstanceSegmentation results
     */
    virtual std::vector<InstanceSegmentation> postprocess(
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes,
        const cv::Size& frame_size) = 0;
};

} // namespace vision_core
