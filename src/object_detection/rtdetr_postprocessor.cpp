#include "neuriplo/tasks/object_detection/rtdetr_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace neuriplo_tasks {

RtDetrPostprocessor::RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, const vision::Size& input_size,
                                         float confidence_threshold, const std::vector<std::string>& output_names)
    : model_type_(model_type), input_size_(input_size), confidence_threshold_(confidence_threshold) {
    findOutputIndices(output_names);
}

namespace {

bool holdsInteger(const Tensor& tensor) {
    if (tensor.data.empty())
        return false;
    return std::holds_alternative<int32_t>(tensor.data.front()) || std::holds_alternative<int64_t>(tensor.data.front());
}

bool looksLikeBoxes(const Tensor& tensor) { return tensor.shape.size() == 3 && tensor.shape[2] == 4; }

bool looksLikeScores(const Tensor& tensor) { return tensor.shape.size() == 2 && !holdsInteger(tensor); }

bool looksLikeLabels(const Tensor& tensor) { return tensor.shape.size() == 2 && holdsInteger(tensor); }

} // namespace

void RtDetrPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    scores_idx_ = -1;
    boxes_idx_ = -1;
    labels_idx_ = -1;

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

void RtDetrPostprocessor::resolveOutputIndices(const std::vector<Tensor>& tensors) {
    const auto at = [&tensors](int index) -> const Tensor& { return tensors[static_cast<size_t>(index)]; };

    const bool named_layout_valid = scores_idx_ >= 0 && boxes_idx_ >= 0 && labels_idx_ >= 0 &&
                                    scores_idx_ != boxes_idx_ && scores_idx_ != labels_idx_ &&
                                    boxes_idx_ != labels_idx_ && looksLikeScores(at(scores_idx_)) &&
                                    looksLikeBoxes(at(boxes_idx_)) && looksLikeLabels(at(labels_idx_));
    if (named_layout_valid)
        return;

    // TorchScript exports carry no output names. Recover the layout from the
    // shape and dtype signature: boxes are the only 3-D tensor with a
    // trailing 4, labels are the integer 2-D tensor, scores the float one.
    int scores = -1;
    int boxes = -1;
    int labels = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto index = static_cast<int>(i);
        if (boxes < 0 && looksLikeBoxes(tensors[i])) {
            boxes = index;
        } else if (labels < 0 && looksLikeLabels(tensors[i])) {
            labels = index;
        } else if (scores < 0 && looksLikeScores(tensors[i])) {
            scores = index;
        }
    }

    if (scores < 0 || boxes < 0 || labels < 0)
        return;

    const Tensor& score_tensor = at(scores);
    const Tensor& box_tensor = at(boxes);
    const Tensor& label_tensor = at(labels);
    if (score_tensor.shape[1] != box_tensor.shape[1] || score_tensor.shape[1] != label_tensor.shape[1])
        return;

    scores_idx_ = scores;
    boxes_idx_ = boxes;
    labels_idx_ = labels;
}

std::vector<Detection> RtDetrPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                        const vision::Size& frame_size) {

    std::vector<Detection> detections;

    switch (model_type_) {
    case ObjectDetectionTask::ModelType::RT_DETR_STYLE: {
        if (tensors.size() < 3) {
            throw std::runtime_error("RT-DETR style models require 3 output tensors");
        }
        resolveOutputIndices(tensors);
        if (scores_idx_ < 0 || boxes_idx_ < 0 || labels_idx_ < 0) {
            // Legacy scores/boxes/labels tuple order.
            scores_idx_ = 0;
            boxes_idx_ = 1;
            labels_idx_ = 2;
        }
        detections =
            postprocessRTDETR(tensors[static_cast<size_t>(scores_idx_)], tensors[static_cast<size_t>(boxes_idx_)],
                              tensors[static_cast<size_t>(labels_idx_)], frame_size);
        break;
    }

    case ObjectDetectionTask::ModelType::RT_DETR_UL: {
        if (tensors.empty()) {
            throw std::runtime_error("RT-DETR UL requires at least 1 output tensor");
        }
        detections = postprocessRTDETRUL(tensors[0], frame_size);
        break;
    }

    default:
        throw std::runtime_error("Unsupported model type for RtDetrPostprocessor");
    }

    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETR(const Tensor& scores, const Tensor& boxes,
                                                              const Tensor& labels, const vision::Size& frame_size) {

    std::vector<Detection> detections;

    if (scores.shape.size() < 2 || boxes.shape.size() < 3) {
        return {};
    }

    const int batch = static_cast<int>(scores.shape[0]);
    const int num_dets = static_cast<int>(scores.shape[1]);
    const size_t batch_scores_stride = static_cast<size_t>(num_dets);
    const size_t batch_boxes_stride = static_cast<size_t>(num_dets) * 4;
    const size_t batch_labels_stride = static_cast<size_t>(num_dets);

    float r_w = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
    float r_h = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);

    for (int b = 0; b < batch; ++b) {
        const size_t scores_batch_offset = static_cast<size_t>(b) * batch_scores_stride;
        const size_t boxes_batch_offset = static_cast<size_t>(b) * batch_boxes_stride;
        const size_t labels_batch_offset = static_cast<size_t>(b) * batch_labels_stride;

        for (int i = 0; i < num_dets; ++i) {
            float score = tensorElementToFloat(scores.data[scores_batch_offset + static_cast<size_t>(i)]);

            if (score < confidence_threshold_)
                continue;

            int class_id = tensorElementToInt(labels.data[labels_batch_offset + static_cast<size_t>(i)]);
            if (class_id < 0)
                continue;

            float x1 = tensorElementToFloat(boxes.data[boxes_batch_offset + static_cast<size_t>(i * 4 + 0)]) * r_w;
            float y1 = tensorElementToFloat(boxes.data[boxes_batch_offset + static_cast<size_t>(i * 4 + 1)]) * r_h;
            float x2 = tensorElementToFloat(boxes.data[boxes_batch_offset + static_cast<size_t>(i * 4 + 2)]) * r_w;
            float y2 = tensorElementToFloat(boxes.data[boxes_batch_offset + static_cast<size_t>(i * 4 + 3)]) * r_h;

            Detection det;
            det.class_id = static_cast<float>(class_id);
            det.class_confidence = score;
            det.bbox = vision::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1),
                                    static_cast<int>(y2 - y1));
            detections.push_back(det);
        }
    }

    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETRUL(const Tensor& output, const vision::Size& frame_size) {

    std::vector<Detection> detections;

    if (output.shape.size() < 3)
        return {};

    const int batch = static_cast<int>(output.shape[0]);
    const int num_dets = static_cast<int>(output.shape[1]);
    const int dims = static_cast<int>(output.shape[2]);
    // Current Ultralytics exports contain normalized cx, cy, w, h, score, class_id.
    if (dims == 6) {
        const size_t count = static_cast<size_t>(batch) * static_cast<size_t>(num_dets);
        if (output.data.size() != count * 6)
            return {};
        const float frame_width = static_cast<float>(frame_size.width);
        const float frame_height = static_cast<float>(frame_size.height);
        for (size_t i = 0; i < count; ++i) {
            const size_t offset = i * 6;
            const float score = tensorElementToFloat(output.data[offset + 4]);
            if (!std::isfinite(score) || score < confidence_threshold_ || score > 1.0f)
                continue;
            const float cx = tensorElementToFloat(output.data[offset]);
            const float cy = tensorElementToFloat(output.data[offset + 1]);
            const float w = tensorElementToFloat(output.data[offset + 2]);
            const float h = tensorElementToFloat(output.data[offset + 3]);
            const float label = tensorElementToFloat(output.data[offset + 5]);
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) ||
                !std::isfinite(label) || w <= 0 || h <= 0 || label < 0 || std::floor(label) != label)
                continue;
            Detection det;
            det.class_id = label;
            det.class_confidence = score;
            det.bbox = vision::Rect(static_cast<int>((cx - w * 0.5f) * frame_width),
                                    static_cast<int>((cy - h * 0.5f) * frame_height), static_cast<int>(w * frame_width),
                                    static_cast<int>(h * frame_height));
            detections.push_back(det);
        }
        return detections;
    }
    const int num_classes = dims - 4;

    if (num_classes <= 0)
        return {};

    const size_t batch_stride = static_cast<size_t>(num_dets) * static_cast<size_t>(dims);

    float r_w = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
    float r_h = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);

    for (int b = 0; b < batch; ++b) {
        const size_t batch_offset = static_cast<size_t>(b) * batch_stride;

        for (int i = 0; i < num_dets; ++i) {
            size_t offset = batch_offset + static_cast<size_t>(i) * static_cast<size_t>(dims);

            float max_score = 0.0f;
            int class_id = -1;
            for (int c = 0; c < num_classes; ++c) {
                float score = tensorElementToFloat(output.data[offset + 4 + static_cast<size_t>(c)]);
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < confidence_threshold_)
                continue;

            float x1 = tensorElementToFloat(output.data[offset + 0]) * r_w;
            float y1 = tensorElementToFloat(output.data[offset + 1]) * r_h;
            float x2 = tensorElementToFloat(output.data[offset + 2]) * r_w;
            float y2 = tensorElementToFloat(output.data[offset + 3]) * r_h;

            Detection det;
            det.class_id = static_cast<float>(class_id);
            det.class_confidence = max_score;
            det.bbox = vision::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1),
                                    static_cast<int>(y2 - y1));
            detections.push_back(det);
        }
    }

    return detections;
}

} // namespace neuriplo_tasks
