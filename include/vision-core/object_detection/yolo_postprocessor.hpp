#pragma once

#include "vision-core/object_detection/object_detection_task.hpp" // For ModelType enum
#include "vision-core/object_detection/postprocessor.hpp"

namespace vision_core {

class YoloPostprocessor : public Postprocessor {
public:
    YoloPostprocessor(ObjectDetectionTask::ModelType model_type, 
                      const cv::Size& input_size,
                      float confidence_threshold, 
                      float nms_threshold);

    std::vector<Detection> postprocess(
        const std::vector<Tensor>& tensors,
        const cv::Size& frame_size) override;

private:
    ObjectDetectionTask::ModelType model_type_;
  cv::Size input_size_; // Model input dimensions from ModelInfo
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

    std::vector<Detection> postprocessYoloV4(
        const std::vector<Tensor>& tensors,
        const cv::Size& frame_size);    

    std::vector<Detection> postprocessYoloNmsFree(
        const Tensor& output,
        const cv::Size& frame_size);

    std::vector<Detection> postprocessYoloNAS(
        const Tensor& boxes,
        const Tensor& scores,
        const cv::Size& frame_size);
    
    std::vector<Detection> postprocessYoloV7E2E(
        const Tensor& num_dets,
        const Tensor& boxes,
        const Tensor& scores,
        const Tensor& classes,
        const cv::Size& frame_size);
        
    void applyNMS(std::vector<Detection>& detections);
    float getTensorFloat(const TensorElement& element);
};

} // namespace vision_core
