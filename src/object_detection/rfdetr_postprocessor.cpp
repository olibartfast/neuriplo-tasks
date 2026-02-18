#include "vision-core/object_detection/rfdetr_postprocessor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace vision_core {

RfDetrPostprocessor::RfDetrPostprocessor(const cv::Size& input_size, float confidence_threshold,
                                         const std::vector<std::string>& output_names)
    : input_size_(input_size), confidence_threshold_(confidence_threshold) {
    findOutputIndices(output_names);
}

void RfDetrPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    // RF-DETR default: dets at 0, labels at 1
    dets_idx_ = 0;
    labels_idx_ = 1;

    if (output_names.empty()) {
        return;
    }

    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& name = output_names[i];
        if (name == "dets") {
            dets_idx_ = static_cast<int>(i);
        } else if (name == "labels") {
            labels_idx_ = static_cast<int>(i);
        }
    }
}

std::vector<Detection> RfDetrPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                        const cv::Size& frame_size) {

    if (tensors.size() < 2) {
        throw std::runtime_error("RF-DETR requires 2 output tensors (dets, labels)");
    }

    const auto& boxes = tensors[static_cast<size_t>(dets_idx_)].data;
    const auto& logits = tensors[static_cast<size_t>(labels_idx_)].data;
    const auto& box_shape = tensors[static_cast<size_t>(dets_idx_)].shape;
    const auto& logit_shape = tensors[static_cast<size_t>(labels_idx_)].shape;

    std::vector<Detection> detections;

    // RF-DETR outputs: dets [1, 300, 4] and labels [1, 300, num_classes] as logits
    if (box_shape.size() < 3 || logit_shape.size() < 3) {
        return {};
    }

    int num_dets = static_cast<int>(box_shape[1]);
    int num_classes = static_cast<int>(logit_shape[2]);

    // Scale factors from input size to frame size
    float scale_w = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
    float scale_h = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);

    for (int i = 0; i < num_dets; ++i) {
        float max_score = 0.0f;
        int max_class_idx = -1;

        // Find max class score applying sigmoid to logits
        for (int c = 0; c < num_classes; ++c) {
            float logit = getTensorFloat(logits[static_cast<size_t>(i * num_classes + c)]);
            float score = 1.0f / (1.0f + std::exp(-logit));
            if (score > max_score) {
                max_score = score;
                max_class_idx = c;
            }
        }

        if (max_score < confidence_threshold_)
            continue;

        // RF-DETR class indices are 1-based
        max_class_idx -= 1;
        if (max_class_idx < 0)
            continue;

        // Boxes are normalized cx,cy,w,h format
        float cx = getTensorFloat(boxes[static_cast<size_t>(i * 4 + 0)]) * static_cast<float>(input_size_.width);
        float cy = getTensorFloat(boxes[static_cast<size_t>(i * 4 + 1)]) * static_cast<float>(input_size_.height);
        float w = getTensorFloat(boxes[static_cast<size_t>(i * 4 + 2)]) * static_cast<float>(input_size_.width);
        float h = getTensorFloat(boxes[static_cast<size_t>(i * 4 + 3)]) * static_cast<float>(input_size_.height);

        // Convert to x,y,w,h and scale to frame size
        float x = (cx - w / 2.0f) * scale_w;
        float y = (cy - h / 2.0f) * scale_h;
        float width = w * scale_w;
        float height = h * scale_h;

        Detection det;
        det.class_id = static_cast<float>(max_class_idx);
        det.class_confidence = max_score;
        det.bbox =
            cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height));
        detections.push_back(det);
    }

    return detections;
}

float RfDetrPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

} // namespace vision_core
