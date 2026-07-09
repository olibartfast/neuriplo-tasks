#include "neuriplo/tasks/core/task_factory.hpp"

#include "neuriplo/tasks/classification/classification_postprocessor.hpp"
#include "neuriplo/tasks/classification/classification_task.hpp"
#include "neuriplo/tasks/depth_estimation/depth_estimation_task.hpp"
#include "neuriplo/tasks/gaussian_splatting/gaussian_splatting_task.hpp"
#include "neuriplo/tasks/image_understanding/image_understanding_task.hpp"
#include "neuriplo/tasks/instance_segmentation/instance_segmentation_task.hpp"
#include "neuriplo/tasks/instance_segmentation/segmentation_postprocessor.hpp"
#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/open_vocab_detection/open_vocab_detection_task.hpp"
#include "neuriplo/tasks/optical_flow/optical_flow_postprocessor.hpp"
#include "neuriplo/tasks/optical_flow/optical_flow_task.hpp"
#include "neuriplo/tasks/pose_estimation/pose_estimation_task.hpp"
#include "neuriplo/tasks/video_classification/video_classification_task.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace neuriplo_tasks {

namespace {

using TaskMatcher = std::function<bool(const std::string&)>;
using TaskCreator = std::function<std::unique_ptr<TaskInterface>(const std::string&, const std::string&,
                                                                 const ModelInfo&, const TaskConfig&)>;

enum class TaskFamily : uint8_t {
    Detection,
    InstanceSegmentation,
    PoseEstimation,
    OpenVocabDetection,
    Classification,
    DepthEstimation,
    GaussianSplatting,
    VideoClassification,
    OpticalFlow,
    ImageUnderstanding,
};

struct TaskDescriptor {
    const char* name;
    TaskFamily family;
    TaskMatcher matches;
    TaskCreator create;
};

bool startsWith(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool contains(const std::string& value, const std::string& needle) { return value.find(needle) != std::string::npos; }

std::unique_ptr<TaskInterface> createSegmentationTask([[maybe_unused]] const std::string& model_type,
                                                      const std::string& normalized, const ModelInfo& model_info,
                                                      const TaskConfig& config) {
    return std::make_unique<InstanceSegmentationTask>(model_info, normalized, config.confidence_threshold,
                                                      config.nms_threshold, config.mask_threshold);
}

std::unique_ptr<TaskInterface> createPoseTask([[maybe_unused]] const std::string& model_type,
                                              const std::string& normalized, const ModelInfo& model_info,
                                              const TaskConfig& config) {
    return std::make_unique<PoseEstimationTask>(model_info, normalized, config.confidence_threshold,
                                                config.nms_threshold);
}

std::unique_ptr<TaskInterface> createDetectionTask([[maybe_unused]] const std::string& model_type,
                                                   const std::string& normalized, const ModelInfo& model_info,
                                                   const TaskConfig& config) {
    return std::make_unique<ObjectDetectionTask>(model_info, normalized, config.confidence_threshold,
                                                 config.nms_threshold);
}

const std::vector<TaskDescriptor>& taskDescriptors() {
    static const std::vector<TaskDescriptor> descriptors = {
        // Instance segmentation before generic YOLO / EdgeCrafter detection.
        {"YoloSegmentation", TaskFamily::InstanceSegmentation,
         [](const std::string& normalized) {
             return normalized == "yoloseg" || normalized == "yolov10seg" || normalized == "yolo26seg" ||
                    normalized == "rfdetrseg" || (startsWith(normalized, "yolo") && contains(normalized, "seg"));
         },
         createSegmentationTask},
        {"EdgeCrafterSegmentation", TaskFamily::InstanceSegmentation,
         [](const std::string& normalized) { return startsWith(normalized, "ecseg"); }, createSegmentationTask},

        // Pose before generic YOLO / EdgeCrafter detection.
        {"RfDetrPose", TaskFamily::PoseEstimation,
         [](const std::string& normalized) {
             return startsWith(normalized, "rfdetr") &&
                    (contains(normalized, "pose") || contains(normalized, "keypoint") || contains(normalized, "kpt"));
         },
         createPoseTask},
        {"YoloPose", TaskFamily::PoseEstimation,
         [](const std::string& normalized) { return startsWith(normalized, "yolo") && contains(normalized, "pose"); },
         createPoseTask},
        {"EdgeCrafterPose", TaskFamily::PoseEstimation,
         [](const std::string& normalized) { return startsWith(normalized, "ecpose"); }, createPoseTask},
        {"VitPose", TaskFamily::PoseEstimation, [](const std::string& normalized) { return normalized == "vitpose"; },
         createPoseTask},

        // Object detection aliases.
        {"YoloDetection", TaskFamily::Detection,
         [](const std::string& normalized) { return startsWith(normalized, "yolo"); }, createDetectionTask},
        {"EdgeCrafterDetection", TaskFamily::Detection,
         [](const std::string& normalized) { return startsWith(normalized, "ecdet"); }, createDetectionTask},
        {"EdgeCrafterFamily", TaskFamily::Detection,
         [](const std::string& normalized) { return startsWith(normalized, "edgecrafter"); },
         [](const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig& config) {
             if (contains(normalized, "seg")) {
                 return createSegmentationTask(model_type, normalized, model_info, config);
             }
             if (contains(normalized, "pose")) {
                 return createPoseTask(model_type, normalized, model_info, config);
             }
             return createDetectionTask(model_type, normalized, model_info, config);
         }},
        {"DetrDetection", TaskFamily::Detection,
         [](const std::string& normalized) {
             return normalized == "rtdetr" || normalized == "rtdetrul" || normalized == "rtdetrultralytics" ||
                    normalized == "rfdetr";
         },
         createDetectionTask},

        // Open-vocabulary detection aliases.
        {"OpenVocabDetection", TaskFamily::OpenVocabDetection,
         [](const std::string& normalized) {
             return normalized == "owlv2" || normalized == "owlvit" || normalized == "groundingdino";
         },
         []([[maybe_unused]] const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig& config) {
             return std::make_unique<OpenVocabDetectionTask>(model_info, normalized, config);
         }},

        // Classification aliases.
        {"Classification", TaskFamily::Classification,
         [](const std::string& normalized) {
             return normalized == "torchvisionclassifier" || normalized == "tensorflowclassifier" ||
                    normalized == "vitclassifier" || startsWith(normalized, "resnet") ||
                    contains(normalized, "tensorflow");
         },
         []([[maybe_unused]] const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig& config) {
             return std::make_unique<ClassificationTask>(model_info, normalized, config.top_k, config.apply_softmax);
         }},

        // Depth estimation aliases.
        {"DepthEstimation", TaskFamily::DepthEstimation,
         [](const std::string& normalized) { return contains(normalized, "depthanythingv2"); },
         []([[maybe_unused]] const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig&) { return std::make_unique<DepthEstimationTask>(model_info, normalized); }},

        // Gaussian splatting aliases.
        {"GaussianSplatting", TaskFamily::GaussianSplatting,
         [](const std::string& normalized) {
             return normalized == "lgm" || normalized == "grm" || normalized == "gaussiansplatting" ||
                    normalized == "lgmmini" || contains(normalized, "splat");
         },
         []([[maybe_unused]] const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig&) { return std::make_unique<GaussianSplattingTask>(model_info, normalized); }},

        // Video classification aliases.
        {"VideoClassification", TaskFamily::VideoClassification,
         [](const std::string& normalized) {
             return normalized == "videomae" || normalized == "vivit" || normalized == "timesformer";
         },
         []([[maybe_unused]] const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig& config) {
             return std::make_unique<VideoClassificationTask>(model_info, normalized, config.top_k,
                                                              config.apply_softmax);
         }},

        // Optical flow aliases.
        {"OpticalFlow", TaskFamily::OpticalFlow, [](const std::string& normalized) { return normalized == "raft"; },
         []([[maybe_unused]] const std::string& model_type, const std::string& normalized, const ModelInfo& model_info,
            const TaskConfig&) { return std::make_unique<OpticalFlowTask>(model_info, normalized); }},

        // vision::Image understanding aliases.
        {"ImageUnderstanding", TaskFamily::ImageUnderstanding,
         [](const std::string& normalized) {
             return normalized == "gemma4" || normalized == "gemma" || normalized == "llama" ||
                    normalized == "llamacpp" || normalized == "imageunderstanding";
         },
         []([[maybe_unused]] const std::string& model_type, [[maybe_unused]] const std::string& normalized,
            const ModelInfo& model_info, const TaskConfig& config) {
             return std::make_unique<ImageUnderstandingTask>(model_info, model_type, config);
         }},
    };

    return descriptors;
}

} // namespace

void TaskFactory::validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes) {
    if (input_sizes.empty()) {
        throw InputDimensionError("Input sizes vector is empty");
    }
    for (const auto& size : input_sizes) {
        if (size.empty()) {
            throw InputDimensionError("An input size vector is empty");
        }
        // -1 is a valid dynamic-dimension marker (used by text/LLM backends).
        if (std::any_of(size.begin(), size.end(), [](int64_t s) { return s <= 0 && s != -1; })) {
            throw InputDimensionError("Non-positive input size detected");
        }
    }
}

std::string TaskFactory::normalizeModelType(const std::string& model_type) {
    std::string normalized;
    normalized.reserve(model_type.size());

    for (char c : model_type) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        if (c == '-' || c == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    return normalized;
}

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(const std::string& model_type,
                                                               const ModelInfo& model_info, const TaskConfig& config) {

    validateInputSizes(model_info.input_shapes);

    const auto normalized = normalizeModelType(model_type);

    if (normalized.empty()) {
        throw std::invalid_argument("Model type string is empty");
    }

    for (const auto& descriptor : taskDescriptors()) {
        if (descriptor.matches(normalized)) {
            return descriptor.create(model_type, normalized, model_info, config);
        }
    }

    throw std::invalid_argument("Unrecognized model type: " + model_type);
}

} // namespace neuriplo_tasks
