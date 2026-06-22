#include "neuriplo/tasks/object_detection/edgecrafter_postprocessor.hpp"

#include "neuriplo/tasks/core/opencv_interop.hpp"
#include "neuriplo/tasks/core/output_name_utils.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

EdgeCrafterPostprocessor::EdgeCrafterPostprocessor(float confidence_threshold,
                                                   const std::vector<std::string>& output_names)
    : confidence_threshold_(confidence_threshold) {
    findOutputIndices(output_names);
}

void EdgeCrafterPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    scores_idx_ = findOutputIndexByName(output_names, "scores", scores_idx_);
    boxes_idx_ = findOutputIndexByName(output_names, "boxes", boxes_idx_);
    labels_idx_ = findOutputIndexByName(output_names, "labels", labels_idx_);
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

    const int batch = static_cast<int>(scores_tensor.shape[0]);
    const int num_dets = static_cast<int>(scores_tensor.shape[1]);
    const size_t batch_scores_stride = static_cast<size_t>(num_dets);
    const size_t batch_boxes_stride = static_cast<size_t>(num_dets) * 4;
    const size_t batch_labels_stride = static_cast<size_t>(num_dets);

    std::vector<Detection> detections;
    detections.reserve(static_cast<size_t>(batch * num_dets));

    for (int b = 0; b < batch; ++b) {
        const size_t scores_batch_offset = static_cast<size_t>(b) * batch_scores_stride;
        const size_t boxes_batch_offset = static_cast<size_t>(b) * batch_boxes_stride;
        const size_t labels_batch_offset = static_cast<size_t>(b) * batch_labels_stride;

        for (int i = 0; i < num_dets; ++i) {
            float score = tensorElementToFloat(scores_tensor.data[scores_batch_offset + static_cast<size_t>(i)]);

            if (score < confidence_threshold_) {
                continue;
            }

            int class_id = tensorElementToInt(labels_tensor.data[labels_batch_offset + static_cast<size_t>(i)]);
            if (class_id < 0) {
                continue;
            }

            const size_t box_offset = boxes_batch_offset + static_cast<size_t>(i) * 4U;
            float x1 = tensorElementToFloat(boxes_tensor.data[box_offset + 0U]);
            float y1 = tensorElementToFloat(boxes_tensor.data[box_offset + 1U]);
            float x2 = tensorElementToFloat(boxes_tensor.data[box_offset + 2U]);
            float y2 = tensorElementToFloat(boxes_tensor.data[box_offset + 3U]);

            Detection det;
            det.class_id = static_cast<float>(class_id);
            det.class_confidence = score;
            det.bbox = fromCvRect(cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                           cv::Point(static_cast<int>(x2), static_cast<int>(y2))));
            detections.push_back(det);
        }
    }

    return detections;
}

} // namespace neuriplo_tasks
