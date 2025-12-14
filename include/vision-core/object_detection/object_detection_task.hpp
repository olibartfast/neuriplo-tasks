#pragma once

#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/preprocessor.hpp"
#include "vision-core/object_detection/postprocessor.hpp"
#include <memory>
#include <string>

namespace vision_core {

/**
 * @brief Unified object detection task for all detection architectures
 * 
 * Handles all detection models with consistent architecture:
 * - YOLO family: v5, v6, v7, v8, v9, v10, v11, v12, NAS
 * - Transformer-based: RT-DETR, RT-DETRv2, RT-DETR Ultralytics, D-FINE, DEIM, RF-DETR
 * 
 * Design principles:
 * 1. Single task class for all detection models
 * 2. Model type enum based on actual postprocessing differences
 * 3. Factory pattern for preprocessor/postprocessor creation
 * 4. Consistent preprocessing and postprocessing interfaces
 */
class ObjectDetectionTask : public TaskInterface {
public:
    /**
     * @brief Detection model types grouped by postprocessing similarity
     */
    enum class ModelType {
        // YOLO family - standard format [batch, proposals, 4+1+classes] or [batch, 4+classes, proposals]
        YOLO_STANDARD,      // YOLOv5, v6, v7, v8, v9, v11, v12
        
        // YOLO variants with different output formats
        YOLO_V10,          // YOLOv10: [batch, detections, 6] - end-to-end, no NMS
        YOLO_NAS,          // YOLO-NAS: separate boxes + scores tensors
        YOLO_V7_E2E,       // YOLOv7 end-to-end: 4 outputs (num_dets, det_boxes, det_scores, det_classes)
        
        // Transformer-based with RT-DETR style postprocessing
        RT_DETR_STYLE,     // RT-DETR, RT-DETRv2, D-FINE, DEIM - same postprocessing
        RT_DETR_UL,        // RT-DETR Ultralytics - different postprocessing 
        RF_DETR            // RF-DETR - different postprocessing
    };

    /**
     * @brief Create detection task for specified model type
     */
    explicit ObjectDetectionTask(const ModelInfo& model_info, 
                                const std::string& model_name,
                                float confidence_threshold = 0.25f,
                                float nms_threshold = 0.45f);

    // TaskInterface implementation
    TaskType getTaskType() override { return TaskType::Detection; }
    
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;
    
    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override;

private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<Postprocessor> postprocessor_;
    float confidence_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;

    /**
     * @brief Detect model type from model name string
     */
    static ModelType detectModelType(const std::string& model_name);
    
    /**
     * @brief Create appropriate preprocessor for model type
     */
    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);
    
    /**
     * @brief Create appropriate postprocessor for model type
     */
    std::unique_ptr<Postprocessor> createPostprocessor(ModelType type, const cv::Size& input_size);

    /**
     * @brief Extract input dimensions from model info
     */
    cv::Size extractInputSize(const ModelInfo& model_info);
    
    /**
     * @brief Validate model-specific tensor requirements
     */
    bool validateTensorInputs(const std::vector<std::vector<TensorElement>>& infer_results,
                             const std::vector<std::vector<int64_t>>& infer_shapes) const;
};

} // namespace vision_core