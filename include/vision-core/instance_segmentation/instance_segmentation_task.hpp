#pragma once
#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_interface.hpp"

#include <memory>
#include <string>

namespace vision_core {

class SegmentationPostprocessor;

class InstanceSegmentationTask : public TaskInterface {
  public:
    enum class ModelType : uint8_t {
        YOLO_SEG,     // YOLO-based segmentation (2 outputs: detections + mask prototypes)
        YOLO_V10_SEG, // YOLOv10 segmentation (NMS-free, 2 outputs: detections+coeffs + mask prototypes)
        YOLO_26_SEG,  // YOLO26 segmentation (NMS-free, 2 outputs: detections+coeffs + mask prototypes)
        RF_DETR_SEG   // RF-DETR segmentation (3 outputs: boxes + labels + masks)
    };

    explicit InstanceSegmentationTask(const ModelInfo& model_info, const std::string& model_name,
                                      float confidence_threshold = 0.25f, float nms_threshold = 0.45f,
                                      float mask_threshold = 0.5f);

    // TaskInterface implementation
    TaskType getTaskType() override { return TaskType::InstanceSegmentation; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<SegmentationPostprocessor> postprocessor_;
    float confidence_threshold_;
    float nms_threshold_;
    float mask_threshold_;

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);
    std::unique_ptr<SegmentationPostprocessor> createPostprocessor(ModelType type);

    cv::Size extractInputSize(const ModelInfo& model_info);

    [[nodiscard]] bool validateTensorInputs(const std::vector<Tensor>& tensors) const;
};

} // namespace vision_core
