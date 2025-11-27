#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/object_detection/yolo_nas_postprocessor.hpp"
#include <memory>

namespace vision_core {

/**
 * @brief YOLO-NAS object detection task implementation
 * 
 * YOLO-NAS models output separated boxes and scores tensors
 */
class YoloNasDetectionTask : public TaskInterface {
public:
    explicit YoloNasDetectionTask(const ModelInfo& model_info)
        : TaskInterface(model_info),
          preprocessor_(std::make_unique<YoloNasPreprocessor>(cv::Size(input_width_, input_height_)))
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
        
        if (infer_results.size() < 2 || infer_shapes.size() < 2) {
            return results;
        }

        // YOLO-NAS has two outputs:
        // Output 0: boxes [batch, num_queries, 4]
        // Output 1: scores [batch, num_queries, num_classes]
        const auto& bbox_data = infer_results[0];
        const auto& score_data = infer_results[1];
        const auto& bbox_shape = infer_shapes[0];
        const auto& score_shape = infer_shapes[1];

        const TensorElement* bbox_ptr = bbox_data.data();
        const TensorElement* score_ptr = score_data.data();
        
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;
        
        auto detections = YoloNasPostprocessor::postprocess(
            bbox_ptr,
            score_ptr,
            bbox_shape,
            score_shape,
            frame_size,
            input_width_,
            input_height_,
            conf_threshold,
            nms_threshold
        );

        for (auto& det : detections) {
            results.push_back(std::move(det));
        }

        return results;
    }

private:
    std::unique_ptr<YoloNasPreprocessor> preprocessor_;
};

// Static registration of YOLO-NAS task variants
namespace {
    struct YoloNasTaskRegistrar {
        YoloNasTaskRegistrar() {
            auto creator = [](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
                return std::make_unique<YoloNasDetectionTask>(info);
            };

            TaskFactory::registerTask("yolonas", creator);
            TaskFactory::registerTask("yolo-nas", creator);
        }
    };
    
    static YoloNasTaskRegistrar yolonas_registrar;
}

} // namespace vision_core
