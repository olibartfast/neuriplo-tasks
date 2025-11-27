#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include "vision-core/object_detection/dfine_postprocessor.hpp"
#include "vision-core/object_detection/rfdetr_postprocessor.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace vision_core {

/**
 * @brief Unified transformer-based detection task
 * 
 * Handles all transformer-based detection models:
 * - RT-DETR, RT-DETRv2
 * - D-FINE
 * - RF-DETR
 * 
 * All these models expect 2 output tensors: boxes and scores.
 * The specific postprocessor is selected based on model name.
 */
class TransformerDetectionTask : public TaskInterface {
public:
    enum class TransformerModel {
        RT_DETR,
        D_FINE,
        RF_DETR
    };

private:
    std::unique_ptr<Preprocessor> preprocessor_;
    float confidence_threshold_;
    TransformerModel model_type_;

    /**
     * @brief Detect model type from model name
     */
    static TransformerModel detectModelType(const std::string& model_type) {
        std::string lower_type = model_type;
        std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
        
        if (lower_type.find("dfine") != std::string::npos ||
            lower_type.find("d-fine") != std::string::npos) {
            return TransformerModel::D_FINE;
        }
        
        if (lower_type.find("rfdetr") != std::string::npos ||
            lower_type.find("rf-detr") != std::string::npos ||
            lower_type.find("rf_detr") != std::string::npos) {
            return TransformerModel::RF_DETR;
        }
        
        // Default to RT-DETR
        return TransformerModel::RT_DETR;
    }

public:
    explicit TransformerDetectionTask(const ModelInfo& model_info,
                                      const std::string& model_type = "",
                                      float confidence_threshold = 0.5f)
        : TaskInterface(model_info)
        , confidence_threshold_(confidence_threshold)
        , model_type_(detectModelType(model_type)) {
        
        if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ != 3) {
            throw InputDimensionError("Invalid transformer detection model dimensions: expected 3-channel input");
        }
        
        // Use appropriate preprocessor based on model type
        switch (model_type_) {
            case TransformerModel::D_FINE:
                preprocessor_ = std::make_unique<DFinePreprocessor>(cv::Size(input_width_, input_height_));
                break;
            case TransformerModel::RF_DETR:
                preprocessor_ = std::make_unique<RfDetrPreprocessor>(cv::Size(input_width_, input_height_));
                break;
            case TransformerModel::RT_DETR:
            default:
                preprocessor_ = std::make_unique<RtDetrPreprocessor>(cv::Size(input_width_, input_height_));
                break;
        }
    }

    TaskType getTaskType() override {
        return TaskType::Detection;
    }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
        std::vector<std::vector<uint8_t>> results;
        results.reserve(imgs.size());
        
        for (const auto& img : imgs) {
            if (img.empty()) {
                throw std::invalid_argument("Empty input image provided");
            }
            results.push_back(preprocessor_->preprocess(img));
        }
        
        return results;
    }

    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override {
        
        if (infer_results.size() < 2 || infer_shapes.size() < 2) {
            throw std::invalid_argument("Transformer detection models require 2 output tensors (boxes and scores)");
        }

        if (infer_results.size() != infer_shapes.size()) {
            throw std::invalid_argument("Mismatch between inference results and shapes");
        }

        std::vector<Result> results;
        
        const TensorElement* bbox_data = infer_results[0].data();
        const TensorElement* score_data = infer_results[1].data();
        
        std::vector<Detection> detections;
        
        // Route to appropriate postprocessor
        switch (model_type_) {
            case TransformerModel::D_FINE:
                detections = DFinePostprocessor::postprocess(
                    bbox_data, score_data,
                    infer_shapes[0], infer_shapes[1],
                    frame_size, input_width_, input_height_,
                    confidence_threshold_
                );
                break;
                
            case TransformerModel::RF_DETR:
                detections = RfDetrPostprocessor::postprocess(
                    bbox_data, score_data,
                    infer_shapes[0], infer_shapes[1],
                    frame_size, input_width_, input_height_,
                    confidence_threshold_
                );
                break;
                
            case TransformerModel::RT_DETR:
            default:
                detections = RtDetrPostprocessor::postprocess(
                    bbox_data, score_data,
                    infer_shapes[0], infer_shapes[1],
                    frame_size, input_width_, input_height_,
                    confidence_threshold_
                );
                break;
        }
        
        for (const auto& detection : detections) {
            results.emplace_back(detection);
        }
        
        return results;
    }
};

// Static registration of transformer detection task variants
namespace {
    struct TransformerDetectionTaskRegistrar {
        TransformerDetectionTaskRegistrar() {
            // RT-DETR variants
            std::vector<std::string> rtdetr_variants = {
                "rtdetr", "rt-detr", "rtdetrv2", "rt-detrv2", "rt_detr", 
                "rtdetr-l", "rtdetr-x", "rt-detr-l", "rt-detr-x"
            };
            
            // D-FINE variants
            std::vector<std::string> dfine_variants = {
                "dfine", "d-fine", "dfine-s", "dfine-m", "dfine-l", "dfine-x",
                "d-fine-s", "d-fine-m", "d-fine-l", "d-fine-x"
            };
            
            // RF-DETR variants
            std::vector<std::string> rfdetr_variants = {
                "rfdetr", "rf-detr", "rf_detr", "rfdetr-s", "rfdetr-m", "rfdetr-l",
                "rf-detr-s", "rf-detr-m", "rf-detr-l"
            };
            
            // Register all variants
            for (const auto& variant : rtdetr_variants) {
                TaskFactory::registerTask(variant, [variant](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
                    return std::make_unique<TransformerDetectionTask>(model_info, variant);
                });
            }
            
            for (const auto& variant : dfine_variants) {
                TaskFactory::registerTask(variant, [variant](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
                    return std::make_unique<TransformerDetectionTask>(model_info, variant);
                });
            }
            
            for (const auto& variant : rfdetr_variants) {
                TaskFactory::registerTask(variant, [variant](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
                    return std::make_unique<TransformerDetectionTask>(model_info, variant);
                });
            }
        }
    };
    
    // Static instance to ensure registration happens at library load time
    static TransformerDetectionTaskRegistrar transformer_detection_registrar;
}

// Factory function for manual registration
std::unique_ptr<TaskInterface> createTransformerDetectionTask(const ModelInfo& model_info, const std::string& model_type) {
    return std::make_unique<TransformerDetectionTask>(model_info, model_type);
}

} // namespace vision_core