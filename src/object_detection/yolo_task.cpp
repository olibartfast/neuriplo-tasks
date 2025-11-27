#include "base_detection_task.hpp"
#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"

namespace vision_core {

class YoloTask : public BaseDetectionTask {
public:
    explicit YoloTask(const ModelInfo& model_info) : BaseDetectionTask(model_info) {
        // Initialize preprocessor
        // YOLO typically uses 640x640, but we should get it from model_info if possible
        // For now, we'll try to deduce it from input shapes or default to 640
        int width = 640;
        int height = 640;
        
        if (!model_info.input_shapes.empty() && model_info.input_shapes[0].size() >= 3) {
             // Assuming NCHW or NHWC, we can try to find spatial dims
             // This logic is simplified; a robust implementation would parse formats carefully
             // But TaskInterface::initializeInputDimensions helper is private/protected
             // Let's re-implement simple logic or just use 640 for now if complex
             // Actually, let's use the logic from TaskInterface if we can access it, 
             // but it's private.
             // We'll assume the user provides correct input shapes.
             const auto& shape = model_info.input_shapes[0];
             if (model_info.input_formats[0] == "FORMAT_NCHW") {
                 height = shape[2];
                 width = shape[3];
             } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
                 height = shape[1];
                 width = shape[2];
             }
        }

        preprocessor_ = std::make_unique<YoloPreprocessor>(cv::Size(width, height));
        input_width_ = width;
        input_height_ = height;
    }

    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override {
        
        if (infer_results.empty()) {
            return {};
        }

        // YOLO postprocessor expects the primary output at index 0
        // For YOLO-NAS, it might need 2 inputs, but let's handle standard YOLO first
        
        auto detections = YoloPostprocessor::postprocess(
            infer_results[0].data(),
            infer_shapes[0],
            frame_size,
            input_width_,
            input_height_,
            confidence_threshold_,
            nms_threshold_
        );

        std::vector<Result> results;
        results.reserve(detections.size());
        for (auto& det : detections) {
            results.emplace_back(std::move(det));
        }
        return results;
    }
};

// Register the task
namespace {
    // Helper to register multiple YOLO variants
    bool register_yolo_variants() {
        auto creator = [](const ModelInfo& info) { return std::make_unique<YoloTask>(info); };
        TaskFactory::registerTask("yolo", creator);
        TaskFactory::registerTask("yolov5", creator);
        TaskFactory::registerTask("yolov6", creator);
        TaskFactory::registerTask("yolov7", creator);
        TaskFactory::registerTask("yolov8", creator);
        TaskFactory::registerTask("yolov9", creator);
        TaskFactory::registerTask("yolov10", creator); // YoloPostprocessor handles v10 too
        TaskFactory::registerTask("yolo11", creator);
        TaskFactory::registerTask("yolov12", creator);
        return true;
    }
    static bool registered = register_yolo_variants();
}

} // namespace vision_core