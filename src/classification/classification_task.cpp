#include "vision-core/core/task_interface.hpp"
#include "vision-core/classification/classifier_postprocessor.hpp"
#include "vision-core/classification/classification_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"

namespace vision_core {

class ClassificationTask : public TaskInterface {
public:
    ClassificationTask(const ModelInfo& model_info, std::unique_ptr<Preprocessor> preprocessor)
        : TaskInterface(model_info), preprocessor_(std::move(preprocessor)) {
        if (!preprocessor_) {
            throw std::invalid_argument("Preprocessor cannot be null");
        }
    }

    TaskType getTaskType() override {
        return TaskType::Classification;
    }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
        return preprocessor_->preprocess(imgs);
    }

    std::vector<Result> postprocess(
        const cv::Size&, // frame_size unused for classification usually
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override {
        
        if (infer_results.empty()) {
            return {};
        }

        auto classifications = ClassifierPostprocessor::postprocess(
            infer_results[0].data(),
            infer_shapes[0],
            5, // top_k default
            true // apply_softmax default
        );

        std::vector<Result> results;
        results.reserve(classifications.size());
        for (auto& cls : classifications) {
            results.emplace_back(std::move(cls));
        }
        return results;
    }

private:
    std::unique_ptr<Preprocessor> preprocessor_;
};

// Register the tasks
namespace {
    bool register_classification_variants() {
        // Helper to create task with specific preprocessor
        auto register_variant = [](const std::string& name, auto preprocessor_creator) {
            TaskFactory::registerTask(name, [preprocessor_creator](const ModelInfo& info) {
                // Determine input size from model info or default
                int width = 224;
                int height = 224;
                if (!info.input_shapes.empty() && info.input_shapes[0].size() >= 3) {
                    // Simplified logic
                    if (info.input_formats[0] == "FORMAT_NCHW") {
                        height = info.input_shapes[0][2];
                        width = info.input_shapes[0][3];
                    } else {
                        height = info.input_shapes[0][1];
                        width = info.input_shapes[0][2];
                    }
                }
                return std::make_unique<ClassificationTask>(
                    info, 
                    preprocessor_creator(cv::Size(width, height))
                );
            });
        };

        // Register Torchvision style (ImageNet norm)
        register_variant("torchvision-classifier", [](cv::Size s) { 
            return std::make_unique<TorchvisionPreprocessor>(s); 
        });
        register_variant("resnet", [](cv::Size s) { 
            return std::make_unique<TorchvisionPreprocessor>(s); 
        });
        register_variant("resnet50", [](cv::Size s) { 
            return std::make_unique<TorchvisionPreprocessor>(s); 
        });
        register_variant("vit-classifier", [](cv::Size s) { 
            return std::make_unique<ViTPreprocessor>(s); 
        });

        // Register TensorFlow style (no ImageNet norm)
        register_variant("tensorflow-classifier", [](cv::Size s) { 
            return std::make_unique<TensorflowPreprocessor>(s); 
        });

        return true;
    }
    static bool registered = register_classification_variants();
}

} // namespace vision_core