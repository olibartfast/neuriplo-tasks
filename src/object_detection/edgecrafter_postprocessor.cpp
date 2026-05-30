#include "vision-core/object_detection/edgecrafter_postprocessor.hpp"

#include <stdexcept>

namespace vision_core {

EdgeCrafterPostprocessor::EdgeCrafterPostprocessor(float confidence_threshold,
                                                   const std::vector<std::string>& output_names)
    : confidence_threshold_(confidence_threshold) {
    findOutputIndices(output_names);
}

void EdgeCrafterPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    if (output_names.empty()) {
        return;
    }

    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& name = output_names[i];
        if (name == "scores") {
            scores_idx_ = static_cast<int>(i);
        } else if (name == "boxes") {
            boxes_idx_ = static_cast<int>(i);
        } else if (name == "labels") {
            labels_idx_ = static_cast<int>(i);
        }
    }
}

std::vector<Detection> EdgeCrafterPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                             const cv::Size& /*frame_size*/) {

    if (tensors.size() < 3) {
        throw std::runtime_error("EdgeCrafter detection requires 3 output tensors (labels, boxes, scores)");
    }

    const auto& scores_tensor = tensors[static_cast<size_t>(scores_idx_)];
    const auto& boxes_tensor = tensors[static_cast<size_t>(boxes_idx_)];
    const auto& labels_tensor = tensors[static_cast<size_t>(labels_idx_)];

    if (scores_tensor.shape.size() < 2 || boxes_tensor.shape.size() < 3 || labels_tensor.shape.size() < 2) {
        return {};
    }

    int num_dets = static_cast<int>(scores_tensor.shape[1]);

    std::vector<Detection> detections;
    detections.reserve(static_cast<size_t>(num_dets));

    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(scores_tensor.data[static_cast<size_t>(i)]);

        if (score < confidence_threshold_) {
            continue;
        }

        int class_id = getTensorInt(labels_tensor.data[static_cast<size_t>(i)]);
        if (class_id < 0) {
            continue;
        }

        float x1 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 0)]);
        float y1 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 1)]);
        float x2 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 2)]);
        float y2 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 3)]);

        Detection det;
        det.class_id = static_cast<float>(class_id);
        det.class_confidence = score;
        det.bbox = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                            cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        detections.push_back(det);
    }

    return detections;
}

float EdgeCrafterPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

int EdgeCrafterPostprocessor::getTensorInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int { return static_cast<int>(value); }, element);
}

} // namespace vision_core
