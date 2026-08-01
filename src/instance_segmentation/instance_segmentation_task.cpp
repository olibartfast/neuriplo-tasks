#include "neuriplo/tasks/instance_segmentation/instance_segmentation_task.hpp"

#include "neuriplo/tasks/instance_segmentation/edgecrafter_segmentation_postprocessor.hpp"
#include "neuriplo/tasks/instance_segmentation/polygon_conversion.hpp"
#include "neuriplo/tasks/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include "neuriplo/tasks/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace neuriplo_tasks {

class InstanceSegmentationPreprocessStrategy {
  public:
    virtual ~InstanceSegmentationPreprocessStrategy() = default;

    [[nodiscard]] virtual std::vector<std::vector<uint8_t>>
    preprocess(const std::vector<vision::Image>& imgs) const = 0;
};

namespace {

class SingleInputSegmentationPreprocessStrategy final : public InstanceSegmentationPreprocessStrategy {
  public:
    explicit SingleInputSegmentationPreprocessStrategy(std::unique_ptr<Preprocessor> preprocessor)
        : preprocessor_(std::move(preprocessor)) {}

    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess(const std::vector<vision::Image>& imgs) const override {
        std::vector<std::vector<uint8_t>> results;
        results.reserve(imgs.size());

        for (const auto& img : imgs) {
            if (img.empty()) {
                throw std::invalid_argument("Empty input image provided");
            }
            results.push_back(preprocessor_->preprocess(img.view()));
        }

        return results;
    }

  private:
    std::unique_ptr<Preprocessor> preprocessor_;
};

class ModelInputSegmentationPreprocessStrategy final : public InstanceSegmentationPreprocessStrategy {
  public:
    ModelInputSegmentationPreprocessStrategy(std::unique_ptr<Preprocessor> preprocessor, const ModelInfo& model_info)
        : preprocessor_(std::move(preprocessor)), model_info_(model_info) {}

    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess(const std::vector<vision::Image>& imgs) const override {
        std::vector<std::vector<uint8_t>> results;
        results.reserve(model_info_.input_shapes.size());

        for (const auto& img : imgs) {
            if (img.empty()) {
                throw std::invalid_argument("Empty input image provided");
            }
        }

        for (size_t i = 0; i < model_info_.input_names.size(); ++i) {
            const auto& input_name = model_info_.input_names[i];
            const auto& input_shape = model_info_.input_shapes[i];

            if (input_shape.size() >= 3) {
                std::vector<uint8_t> batched;
                for (const auto& img : imgs) {
                    auto buf = preprocessor_->preprocess(img.view());
                    batched.insert(batched.end(), buf.begin(), buf.end());
                }
                results.push_back(std::move(batched));
            } else if (input_name == "orig_target_sizes" || input_name == "orig_size") {
                std::vector<int64_t> all_sizes;
                all_sizes.reserve(imgs.size() * 2);
                for (const auto& img : imgs) {
                    all_sizes.push_back(static_cast<int64_t>(img.cols()));
                    all_sizes.push_back(static_cast<int64_t>(img.rows()));
                }
                const auto* begin = reinterpret_cast<const uint8_t*>(all_sizes.data());
                const auto* end = begin + all_sizes.size() * sizeof(int64_t);
                results.emplace_back(begin, end);
            } else {
                results.emplace_back();
            }
        }

        return results;
    }

  private:
    std::unique_ptr<Preprocessor> preprocessor_;
    ModelInfo model_info_;
};

} // namespace

InstanceSegmentationTask::InstanceSegmentationTask(const ModelInfo& model_info, const std::string& model_name,
                                                   float confidence_threshold, float nms_threshold,
                                                   float mask_threshold, SegmentationOutput segmentation_output)
    : TaskInterface(model_info), model_type_(detectModelType(model_name)), model_name_(model_name),
      confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold), mask_threshold_(mask_threshold),
      segmentation_output_(segmentation_output) {
    vision::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    preprocess_strategy_ = createPreprocessStrategy(model_type_, input_size);
    if (!preprocess_strategy_) {
        throw std::runtime_error("Failed to create preprocessor for segmentation model: " + model_name);
    }

    postprocessor_ = createPostprocessor(model_type_);
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for segmentation model: " + model_name);
    }
}

InstanceSegmentationTask::~InstanceSegmentationTask() = default;

std::vector<std::vector<uint8_t>> InstanceSegmentationTask::preprocess(const std::vector<vision::Image>& imgs) {
    return preprocess_strategy_->preprocess(imgs);
}

std::vector<Result> InstanceSegmentationTask::postprocess(const vision::Size& frame_size,
                                                          const std::vector<Tensor>& tensors) {

    if (!validateTensorInputs(tensors)) {
        return {};
    }

    auto segmentations = postprocessor_->postprocess(tensors, frame_size);

    std::vector<Result> results;
    results.reserve(segmentations.size());
    for (auto& segmentation : segmentations) {
        if (segmentation_output_ == SegmentationOutput::Polygon) {
            segmentation.polygons = maskToPolygons(segmentation.mask);
            segmentation.mask = {};
            segmentation.mask_data.clear();
            segmentation.mask_height = 0;
            segmentation.mask_width = 0;
        }
        results.emplace_back(std::move(segmentation));
    }

    return results;
}

InstanceSegmentationTask::ModelType InstanceSegmentationTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    if (lower_name.find("yolov10") != std::string::npos && lower_name.find("seg") != std::string::npos) {
        return ModelType::YOLO_V10_SEG;
    }

    if (lower_name.find("yolo26") != std::string::npos && lower_name.find("seg") != std::string::npos) {
        return ModelType::YOLO_26_SEG;
    }

    if (lower_name.find("ecseg") == 0 ||
        (lower_name.find("edgecrafter") == 0 && lower_name.find("seg") != std::string::npos)) {
        return ModelType::EDGECRAFTER_SEG;
    }

    if (lower_name.find("rfdetr") != std::string::npos || lower_name.find("rf-detr") != std::string::npos ||
        lower_name.find("rfdetrseg") != std::string::npos) {
        return ModelType::RF_DETR_SEG;
    }

    return ModelType::YOLO_SEG;
}

std::unique_ptr<Preprocessor> InstanceSegmentationTask::createPreprocessor(ModelType type,
                                                                           const vision::Size& input_size) {
    switch (type) {
    case ModelType::YOLO_SEG:
    case ModelType::YOLO_V10_SEG:
    case ModelType::YOLO_26_SEG:
        return std::make_unique<YoloPreprocessor>(input_size);

    case ModelType::RF_DETR_SEG:
    case ModelType::EDGECRAFTER_SEG:
        return std::make_unique<RfDetrPreprocessor>(input_size);

    default:
        return nullptr;
    }
}

std::unique_ptr<InstanceSegmentationPreprocessStrategy>
InstanceSegmentationTask::createPreprocessStrategy(ModelType type, const vision::Size& input_size) {
    auto preprocessor = createPreprocessor(type, input_size);
    if (!preprocessor) {
        return nullptr;
    }

    if (type == ModelType::EDGECRAFTER_SEG) {
        return std::make_unique<ModelInputSegmentationPreprocessStrategy>(std::move(preprocessor), model_info_);
    }

    return std::make_unique<SingleInputSegmentationPreprocessStrategy>(std::move(preprocessor));
}

std::unique_ptr<SegmentationPostprocessor> InstanceSegmentationTask::createPostprocessor(ModelType type) {
    vision::Size input_size(input_width_, input_height_);

    switch (type) {
    case ModelType::YOLO_SEG:
    case ModelType::YOLO_V10_SEG:
    case ModelType::YOLO_26_SEG:
        return std::make_unique<YoloSegmentationPostprocessor>(type, input_size, confidence_threshold_, nms_threshold_,
                                                               mask_threshold_);

    case ModelType::RF_DETR_SEG:
        return std::make_unique<RfDetrSegmentationPostprocessor>(input_size, confidence_threshold_, mask_threshold_,
                                                                 model_info_.output_names);

    case ModelType::EDGECRAFTER_SEG:
        return std::make_unique<EdgeCrafterSegmentationPostprocessor>(confidence_threshold_, mask_threshold_,
                                                                      model_info_.output_names);

    default:
        return nullptr;
    }
}

vision::Size InstanceSegmentationTask::extractInputSize(const ModelInfo& model_info) {
    int width = 640;
    int height = 640;

    if (!model_info.input_shapes.empty() && model_info.input_shapes[0].size() >= 3) {
        const auto& shape = model_info.input_shapes[0];
        if (shape.size() == 4) {
            if (model_info.input_formats[0] == "FORMAT_NCHW") {
                height = static_cast<int>(shape[2]);
                width = static_cast<int>(shape[3]);
            } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            }
        } else if (shape.size() == 3) {
            if (model_info.input_formats[0] == "FORMAT_NCHW") {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
            }
        }
    }

    return vision::Size(width, height);
}

bool InstanceSegmentationTask::validateTensorInputs(const std::vector<Tensor>& tensors) const {

    if (tensors.empty()) {
        return false;
    }

    switch (model_type_) {
    case ModelType::YOLO_SEG:
    case ModelType::YOLO_V10_SEG:
    case ModelType::YOLO_26_SEG:
        return tensors.size() >= 2;

    case ModelType::RF_DETR_SEG:
        return tensors.size() >= 3;

    case ModelType::EDGECRAFTER_SEG:
        return tensors.size() >= 4;

    default:
        return false;
    }
}

} // namespace neuriplo_tasks
