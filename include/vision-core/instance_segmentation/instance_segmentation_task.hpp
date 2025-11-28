#pragma once

#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/preprocessor.hpp"
#include <memory>
#include <string>

namespace vision_core {

/**
 * @brief Unified instance segmentation task for all segmentation architectures
 * 
 * Handles all segmentation model types with consistent architecture:
 * - YOLO Segmentation (YOLOv5-seg, YOLOv8-seg, YOLO11-seg, YOLO12-seg)
 * - RF-DETR Segmentation
 * 
 * Design principles:
 * 1. Single task class for all segmentation models
 * 2. Model type enum based on actual postprocessing differences
 * 3. Factory pattern for preprocessor creation
 * 4. Unified output format with masks and bounding boxes
 */
class InstanceSegmentationTask : public TaskInterface {
public:
    /**
     * @brief Segmentation model types grouped by postprocessing similarity
     */
    enum class ModelType {
        YOLO_SEG,      // YOLO-based segmentation (2 outputs: detections + mask prototypes)
        RF_DETR_SEG    // RF-DETR segmentation (3 outputs: boxes + labels + masks)
    };

    /**
     * @brief Create segmentation task for specified model type
     */
    explicit InstanceSegmentationTask(const ModelInfo& model_info, 
                                    const std::string& model_name,
                                    float confidence_threshold = 0.25f,
                                    float nms_threshold = 0.45f,
                                    float mask_threshold = 0.5f);

    // TaskInterface implementation
    TaskType getTaskType() override { return TaskType::InstanceSegmentation; }
    
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;
    
    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override;

private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    float confidence_threshold_;
    float nms_threshold_;
    float mask_threshold_;
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
     * @brief Extract input dimensions from model info
     */
    cv::Size extractInputSize(const ModelInfo& model_info);
    
    /**
     * @brief Validate model-specific tensor requirements
     */
    bool validateTensorInputs(const std::vector<std::vector<TensorElement>>& infer_results,
                             const std::vector<std::vector<int64_t>>& infer_shapes) const;
    
    // Internal postprocessing methods
    std::vector<InstanceSegmentation> postprocessYoloSeg(
        const std::vector<TensorElement>& detections,
        const std::vector<TensorElement>& mask_protos,
        const std::vector<int64_t>& det_shape,
        const std::vector<int64_t>& mask_shape,
        const cv::Size& frame_size);
    
    std::vector<InstanceSegmentation> postprocessRFDETRSeg(
        const std::vector<TensorElement>& boxes,
        const std::vector<TensorElement>& labels,
        const std::vector<TensorElement>& masks,
        const std::vector<int64_t>& box_shape,
        const std::vector<int64_t>& label_shape,
        const std::vector<int64_t>& mask_shape,
        const cv::Size& frame_size);
    
    // Helper methods
    float getTensorFloat(const TensorElement& element);
    void applyNMS(std::vector<InstanceSegmentation>& segmentations);
};

} // namespace vision_core