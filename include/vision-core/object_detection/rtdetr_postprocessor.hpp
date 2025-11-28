#pragma once

#include "vision-core/object_detection/postprocessor.hpp"
#include "vision-core/object_detection/object_detection_task.hpp"

namespace vision_core {

class RtDetrPostprocessor : public Postprocessor {
public:
    RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, float confidence_threshold);

    std::vector<Detection> postprocess(
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes,
        const cv::Size& frame_size) override;

private:
    ObjectDetectionTask::ModelType model_type_;
    float confidence_threshold_;

    std::vector<Detection> postprocessRTDETR(
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& scores,
        const std::vector<int64_t>& box_shape,
        const std::vector<int64_t>& score_shape,
        const cv::Size& frame_size);

    std::vector<Detection> postprocessRTDETRUL(
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& scores,
        const std::vector<int64_t>& box_shape,
        const std::vector<int64_t>& score_shape,
        const cv::Size& frame_size);

    std::vector<Detection> postprocessRFDETR(
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& scores,
        const std::vector<int64_t>& box_shape,
        const std::vector<int64_t>& score_shape,
        const cv::Size& frame_size);
        
    float getTensorFloat(const TensorElement& element);
};

} // namespace vision_core
