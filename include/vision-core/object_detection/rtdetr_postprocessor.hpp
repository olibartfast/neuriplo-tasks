#pragma once

#include "vision-core/object_detection/postprocessor.hpp"
#include "vision-core/object_detection/object_detection_task.hpp"

namespace vision_core {

/**
 * @brief RT-DETR/DEIM/DFINE postprocessor
 * 
 * These models output 3 tensors:
 * - scores: [1, 300] - confidence per detection
 * - boxes: [1, 300, 4] - x1, y1, x2, y2 coordinates
 * - labels: [1, 300] - class ID per detection
 */
class RtDetrPostprocessor : public Postprocessor {
public:
    RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, 
                        const cv::Size& input_size,
                        float confidence_threshold,
                        const std::vector<std::string>& output_names = {});

    std::vector<Detection> postprocess(
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes,
        const cv::Size& frame_size) override;

private:
    ObjectDetectionTask::ModelType model_type_;
    cv::Size input_size_;
    float confidence_threshold_;
    int scores_idx_{0};
    int boxes_idx_{1};
    int labels_idx_{2};

    void findOutputIndices(const std::vector<std::string>& output_names);
    
    // RT-DETR/DEIM/DFINE: 3 separate outputs
    std::vector<Detection> postprocessRTDETR(
        const std::vector<TensorElement>& scores,
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& labels,
        const std::vector<int64_t>& score_shape,
        const std::vector<int64_t>& box_shape,
        const cv::Size& frame_size);

    // RT-DETR Ultralytics: single combined output
    std::vector<Detection> postprocessRTDETRUL(
        const std::vector<TensorElement>& output,
        const std::vector<int64_t>& output_shape,
        const cv::Size& frame_size);
        
    float getTensorFloat(const TensorElement& element);
    int getTensorInt(const TensorElement& element);
};

} // namespace vision_core
