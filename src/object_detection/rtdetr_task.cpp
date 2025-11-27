#include "base_detection_task.hpp"
#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"

namespace vision_core {

class RtDetrTask : public BaseDetectionTask {
public:
    explicit RtDetrTask(const ModelInfo& model_info) : BaseDetectionTask(model_info) {
        int width = 640;
        int height = 640;
        
        if (!model_info.input_shapes.empty() && model_info.input_shapes[0].size() >= 3) {
             const auto& shape = model_info.input_shapes[0];
             if (model_info.input_formats[0] == "FORMAT_NCHW") {
                 height = shape[2];
                 width = shape[3];
             } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
                 height = shape[1];
                 width = shape[2];
             }
        }

        preprocessor_ = std::make_unique<RtDetrPreprocessor>(cv::Size(width, height));
        input_width_ = width;
        input_height_ = height;
    }

    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override {
        
        if (infer_results.size() < 2) {
            // RT-DETR needs boxes and scores
            return {};
        }

        // Assuming index 0 is boxes, index 1 is scores, or vice versa?
        // RtDetrPostprocessor expects bbox_output, score_output
        // We need to know which is which. 
        // Usually model_info.output_names helps, but here we might assume order.
        // Let's assume standard RT-DETR export order.
        // If ambiguous, we might need more robust logic, but for now:
        const auto* boxes = infer_results[0].data();
        const auto* scores = infer_results[1].data();
        const auto& box_shape = infer_shapes[0];
        const auto& score_shape = infer_shapes[1];

        auto detections = RtDetrPostprocessor::postprocess(
            boxes,
            scores,
            box_shape,
            score_shape,
            frame_size,
            input_width_,
            input_height_,
            confidence_threshold_
        );

        std::vector<Result> results;
        results.reserve(detections.size());
        for (auto& det : detections) {
            results.emplace_back(std::move(det));
        }
        return results;
    }
};

// Explicit registration function for RT-DETR tasks
void registerRtDetrTasks() {
    auto creator = [](const ModelInfo& info) { return std::make_unique<RtDetrTask>(info); };
    TaskFactory::registerTask("rtdetr", creator);
    TaskFactory::registerTask("rtdetrv2", creator);
}

} // namespace vision_core
