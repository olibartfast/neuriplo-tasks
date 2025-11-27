#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include <memory>

namespace vision_core {

/**
 * @brief YOLO object detection task implementation
 * 
 * Supports YOLO v5-v12 models with automatic format detection
 */
class YoloDetectionTask : public TaskInterface {
public:
    explicit YoloDetectionTask(const ModelInfo& model_info)
        : TaskInterface(model_info),
          preprocessor_(std::make_unique<YoloPreprocessor>(cv::Size(input_width_, input_height_)))
    {
    }

    TaskType getTaskType() override {
        return TaskType::Detection;
    }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
        std::vector<std::vector<uint8_t>> preprocessed_data;
        preprocessed_data.reserve(imgs.size());
        
        for (const auto& img : imgs) {
            preprocessed_data.push_back(preprocessor_->preprocess(img));
        }
        
        return preprocessed_data;
    }

    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override 
    {
        std::vector<Result> results;
        
        if (infer_results.empty() || infer_shapes.empty()) {
            return results;
        }

        // YOLO models typically have a single output tensor
        const auto& output_data = infer_results[0];
        const auto& output_shape = infer_shapes[0];

        // Convert TensorElement vector to raw pointer for postprocessor
        const TensorElement* data_ptr = output_data.data();
        
        // Use confidence and NMS thresholds from model_info or defaults
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;
        
        auto detections = YoloPostprocessor::postprocess(
            data_ptr,
            output_shape,
            frame_size,
            input_width_,
            input_height_,
            conf_threshold,
            nms_threshold
        );

        // Convert detections to Result variant
        for (auto& det : detections) {
            results.push_back(std::move(det));
        }

        return results;
    }

private:
    std::unique_ptr<YoloPreprocessor> preprocessor_;
};

// Note: YOLO task registration is handled by yolo_task.cpp
// This class remains for backward compatibility but does not register itself

} // namespace vision_core
