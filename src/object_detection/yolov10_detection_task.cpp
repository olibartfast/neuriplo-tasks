#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/object_detection/yolov10_postprocessor.hpp"
#include <memory>

namespace vision_core {

/**
 * @brief YOLOv10 object detection task implementation
 * 
 * YOLOv10 is an end-to-end detector that doesn't require NMS
 */
class Yolov10DetectionTask : public TaskInterface {
public:
    explicit Yolov10DetectionTask(const ModelInfo& model_info)
        : TaskInterface(model_info),
          preprocessor_(std::make_unique<Yolov10Preprocessor>(cv::Size(input_width_, input_height_)))
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

        // YOLOv10 has a single output tensor: [batch, num_detections, 6]
        const auto& output_data = infer_results[0];
        const auto& output_shape = infer_shapes[0];

        const TensorElement* data_ptr = output_data.data();
        
        float conf_threshold = 0.5f;
        
        auto detections = Yolov10Postprocessor::postprocess(
            data_ptr,
            output_shape,
            frame_size,
            input_width_,
            input_height_,
            conf_threshold
        );

        for (auto& det : detections) {
            results.push_back(std::move(det));
        }

        return results;
    }

private:
    std::unique_ptr<Yolov10Preprocessor> preprocessor_;
};

// Explicit registration function for YOLOv10 tasks
void registerYolov10Tasks() {
    auto creator = [](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
        return std::make_unique<Yolov10DetectionTask>(info);
    };

    TaskFactory::registerTask("yolov10", creator);
}

} // namespace vision_core
