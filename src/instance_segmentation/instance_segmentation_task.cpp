#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include "vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace vision_core {

/**
 * @brief Instance segmentation task implementation
 * 
 * Handles various instance segmentation models including:
 * - YOLO-Seg (YOLOv5-seg, YOLOv8-seg, YOLO11-seg)
 * - RF-DETR Segmentation
 * 
 * Automatically detects model type and routes to appropriate postprocessor.
 */
class InstanceSegmentationTask : public TaskInterface {
public:
    enum class SegmentationModel {
        YOLO_SEG,     // YOLO-based segmentation (requires 2 outputs: detection + mask prototypes)
        RFDETR_SEG    // RF-DETR segmentation (requires 3 outputs: boxes + labels + masks)
    };

private:
    std::unique_ptr<Preprocessor> preprocessor_;
    float confidence_threshold_;
    float nms_threshold_;
    float mask_threshold_;
    SegmentationModel model_type_;

    /**
     * @brief Detect model type from model name
     */
    static SegmentationModel detectModelType(const std::string& model_type) {
        std::string lower_type = model_type;
        std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
        
        if (lower_type.find("rfdetr") != std::string::npos ||
            lower_type.find("rf-detr") != std::string::npos ||
            lower_type.find("rf_detr") != std::string::npos) {
            return SegmentationModel::RFDETR_SEG;
        }
        
        // Default to YOLO segmentation
        return SegmentationModel::YOLO_SEG;
    }

    /**
     * @brief Detect model type from tensor count
     */
    SegmentationModel detectModelFromTensors(const std::vector<std::vector<int64_t>>& shapes) const {
        if (shapes.size() == 3) {
            return SegmentationModel::RFDETR_SEG; // boxes, labels, masks
        }
        if (shapes.size() == 2) {
            return SegmentationModel::YOLO_SEG;   // detection, mask prototypes
        }
        
        return model_type_; // Fall back to name-based detection
    }

public:
    explicit InstanceSegmentationTask(const ModelInfo& model_info,
                                      const std::string& model_type = "",
                                      float confidence_threshold = 0.25f,
                                      float nms_threshold = 0.45f,
                                      float mask_threshold = 0.5f)
        : TaskInterface(model_info)
        , confidence_threshold_(confidence_threshold)
        , nms_threshold_(nms_threshold)
        , mask_threshold_(mask_threshold)
        , model_type_(detectModelType(model_type)) {
        
        if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ != 3) {
            throw InputDimensionError("Invalid segmentation model dimensions: expected 3-channel input");
        }
        
        // Use appropriate preprocessor based on model type
        if (model_type_ == SegmentationModel::YOLO_SEG) {
            preprocessor_ = std::make_unique<YoloPreprocessor>(cv::Size(input_width_, input_height_));
        } else {
            preprocessor_ = std::make_unique<RfDetrPreprocessor>(cv::Size(input_width_, input_height_));
        }
    }

    TaskType getTaskType() override {
        return TaskType::InstanceSegmentation;
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
        
        if (infer_results.empty() || infer_shapes.empty()) {
            return {};
        }

        if (infer_results.size() != infer_shapes.size()) {
            throw std::invalid_argument("Mismatch between inference results and shapes");
        }

        // Auto-detect model type from tensor count
        SegmentationModel runtime_model = detectModelFromTensors(infer_shapes);
        
        std::vector<Result> results;
        
        try {
            if (runtime_model == SegmentationModel::RFDETR_SEG) {
                results = processRfDetrSegmentation(frame_size, infer_results, infer_shapes);
            } else {
                results = processYoloSegmentation(frame_size, infer_results, infer_shapes);
            }
        } catch (const std::exception& e) {
            // Fallback: try other model type
            try {
                if (runtime_model == SegmentationModel::YOLO_SEG) {
                    results = processRfDetrSegmentation(frame_size, infer_results, infer_shapes);
                } else {
                    results = processYoloSegmentation(frame_size, infer_results, infer_shapes);
                }
            } catch (...) {
                throw std::runtime_error("Failed to process segmentation output with any model type: " + std::string(e.what()));
            }
        }
        
        return results;
    }

private:
    std::vector<Result> processYoloSegmentation(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) {
        
        if (infer_results.size() < 2) {
            throw std::invalid_argument("YOLO segmentation requires 2 output tensors (detection + masks)");
        }
        
        std::vector<Result> results;
        
        const TensorElement* detection_data = infer_results[0].data();
        const TensorElement* mask_data = infer_results[1].data();
        
        auto segmentations = YoloSegmentationPostprocessor::postprocess(
            detection_data,
            mask_data,
            infer_shapes[0],
            infer_shapes[1],
            frame_size,
            input_width_,
            input_height_,
            confidence_threshold_,
            nms_threshold_,
            mask_threshold_
        );
        
        for (const auto& segmentation : segmentations) {
            results.emplace_back(segmentation);
        }
        
        return results;
    }

    std::vector<Result> processRfDetrSegmentation(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) {
        
        if (infer_results.size() < 3) {
            throw std::invalid_argument("RF-DETR segmentation requires 3 output tensors (boxes + labels + masks)");
        }
        
        std::vector<Result> results;
        
        const TensorElement* bbox_data = infer_results[0].data();
        const TensorElement* label_data = infer_results[1].data();
        const TensorElement* mask_data = infer_results[2].data();
        
        auto segmentations = RfDetrSegmentationPostprocessor::postprocess(
            bbox_data,
            label_data,
            mask_data,
            infer_shapes[0],
            infer_shapes[1],
            infer_shapes[2],
            frame_size,
            input_width_,
            input_height_,
            confidence_threshold_,
            mask_threshold_
        );
        
        // Convert RfDetr results to standard InstanceSegmentation format
        for (const auto& seg : segmentations) {
            InstanceSegmentation instance_seg;
            instance_seg.bbox = seg.bbox;
            instance_seg.class_confidence = seg.confidence;
            instance_seg.class_id = static_cast<float>(seg.class_id);
            
            // Convert mask to vector format
            if (!seg.mask.empty()) {
                instance_seg.mask_height = seg.mask.rows;
                instance_seg.mask_width = seg.mask.cols;
                
                // Convert mask to uint8_t vector
                instance_seg.mask_data.resize(seg.mask.total());
                if (seg.mask.type() == CV_8UC1) {
                    std::memcpy(instance_seg.mask_data.data(), seg.mask.data, seg.mask.total());
                } else {
                    // Convert to 8-bit if needed
                    cv::Mat mask_8u;
                    seg.mask.convertTo(mask_8u, CV_8UC1, 255.0);
                    std::memcpy(instance_seg.mask_data.data(), mask_8u.data, mask_8u.total());
                }
            }
            
            results.emplace_back(std::move(instance_seg));
        }
        
        return results;
    }
};

// Explicit registration function for instance segmentation tasks
void registerInstanceSegmentationTasks() {
    // YOLO segmentation variants
    std::vector<std::string> yolo_seg_variants = {
        "yoloseg"
    };
    
    // RF-DETR segmentation variants
    std::vector<std::string> rfdetr_seg_variants = {
        "rfdetr"
    };
    
    // Register YOLO segmentation variants
    for (const auto& variant : yolo_seg_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<InstanceSegmentationTask>(model_info, variant);
        });
    }
    
    // Register RF-DETR segmentation variants
    for (const auto& variant : rfdetr_seg_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<InstanceSegmentationTask>(model_info, variant);
        });
    }
}

// Factory function for manual registration
std::unique_ptr<TaskInterface> createInstanceSegmentationTask(const ModelInfo& model_info, const std::string& model_type) {
    return std::make_unique<InstanceSegmentationTask>(model_info, model_type);
}

} // namespace vision_core