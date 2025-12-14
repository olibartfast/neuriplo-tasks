#pragma once

#include "vision-core/object_detection/postprocessor.hpp"
#include "vision-core/object_detection/object_detection_task.hpp" // For ModelType enum

namespace vision_core {

class YoloPostprocessor : public Postprocessor {
public:
    YoloPostprocessor(ObjectDetectionTask::ModelType model_type, 
                      const cv::Size& input_size,
                      float confidence_threshold, 
                      float nms_threshold);

    std::vector<Detection> postprocess(
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes,
        const cv::Size& frame_size) override;

private:
    ObjectDetectionTask::ModelType model_type_;
    cv::Size input_size_;  // Model input dimensions from ModelInfo
    float confidence_threshold_;
    float nms_threshold_;

    /**
     * @brief Convert letterboxed coordinates back to original frame coordinates
     * @param cx Center X in model space
     * @param cy Center Y in model space  
     * @param w Width in model space
     * @param h Height in model space
     * @param frame_size Original frame dimensions
     * @return Bounding box in original frame coordinates
     */
    cv::Rect scaleToOriginal(float cx, float cy, float w, float h, const cv::Size& frame_size) const;

    std::vector<Detection> postprocessYoloStandard(
        const std::vector<TensorElement>& output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size);

    std::vector<Detection> postprocessYoloV10(
        const std::vector<TensorElement>& output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size);

    std::vector<Detection> postprocessYoloNAS(
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& scores,
        const std::vector<int64_t>& box_shape,
        const std::vector<int64_t>& score_shape,
        const cv::Size& frame_size);
    
    std::vector<Detection> postprocessYoloV7E2E(
        const std::vector<TensorElement>& num_dets,
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& scores,
        const std::vector<TensorElement>& classes,
        const std::vector<int64_t>& boxes_shape,
        const cv::Size& frame_size);
        
    void applyNMS(std::vector<Detection>& detections);
    float getTensorFloat(const TensorElement& element);
};

} // namespace vision_core
