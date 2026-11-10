#include "neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>

namespace neuriplo_tasks {

Monocular3DPostprocessor::Monocular3DPostprocessor(const Size& input_size, float confidence_threshold,
                                                   float nms_threshold, bool end_to_end, bool top_first)
    : input_size_(input_size), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold),
      end_to_end_(end_to_end), top_first_(top_first) {}

BoundingBox Monocular3DPostprocessor::scaleBoxToOriginal(float cx, float cy, float w, float h,
                                                         const Size& frame_size) const {
    const float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    const float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    if (r_h > r_w) {
        const float pad_h =
            (static_cast<float>(input_size_.height) - r_w * static_cast<float>(frame_size.height)) / 2.0f;
        x = static_cast<int>((cx - w / 2.0f) / r_w);
        y = static_cast<int>((cy - h / 2.0f - pad_h) / r_w);
        width = static_cast<int>(w / r_w);
        height = static_cast<int>(h / r_w);
    } else {
        const float pad_w = (static_cast<float>(input_size_.width) - r_h * static_cast<float>(frame_size.width)) / 2.0f;
        x = static_cast<int>((cx - w / 2.0f - pad_w) / r_h);
        y = static_cast<int>((cy - h / 2.0f) / r_h);
        width = static_cast<int>(w / r_h);
        height = static_cast<int>(h / r_h);
    }
    return BoundingBox(x, y, width, height);
}

std::pair<float, float> Monocular3DPostprocessor::scalePointToOriginal(float x, float y, const Size& frame_size) const {
    const float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    const float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);
    if (r_h > r_w) {
        const float pad_h =
            (static_cast<float>(input_size_.height) - r_w * static_cast<float>(frame_size.height)) / 2.0f;
        return {x / r_w, (y - pad_h) / r_w};
    }
    const float pad_w = (static_cast<float>(input_size_.width) - r_h * static_cast<float>(frame_size.width)) / 2.0f;
    return {(x - pad_w) / r_h, y / r_h};
}

void Monocular3DPostprocessor::applyNMS(std::vector<Monocular3DDetection>& detections) const {
    if (detections.empty()) {
        return;
    }
    std::vector<image_ops::DetectionBox> boxes;
    boxes.reserve(detections.size());
    for (const auto& det : detections) {
        boxes.push_back({det.bbox, det.class_confidence, det.class_id});
    }
    auto keep = image_ops::nms(boxes, nms_threshold_);
    std::vector<Monocular3DDetection> filtered;
    filtered.reserve(keep.size());
    for (int idx : keep) {
        filtered.push_back(detections[static_cast<size_t>(idx)]);
    }
    detections = std::move(filtered);
}

std::vector<Monocular3DDetection> Monocular3DPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                        const Size& original_size) const {
    if (tensors.empty()) {
        return {};
    }
    const auto& tensor = tensors[0];
    const auto& shape = tensor.shape;
    const auto& data = tensor.data;
    if (shape.size() != 3) {
        return {};
    }
    const int batch = static_cast<int>(shape[0]);
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);
    if (batch <= 0 || dim1 <= 0 || dim2 <= 0) {
        return {};
    }

    // Rows layout ([B, N, C], YOLOv5 pose or YOLO26 end-to-end) vs
    // channels-first ([B, C, N], YOLOv8/v11 pose).
    const bool has_objectness = (dim2 < dim1);
    const bool end_to_end = end_to_end_ && has_objectness && dim2 > 6 && (dim2 - 6) % 3 == 0;
    const int channels = has_objectness ? dim2 : dim1;
    const int anchors = has_objectness ? dim1 : dim2;
    if (anchors <= 0 || channels <= 0) {
        return {};
    }
    const size_t batch_stride = static_cast<size_t>(channels) * static_cast<size_t>(anchors);

    // Eight cuboid corners, two or three values each (x, y, [conf]).
    // v1 exports omit the per-corner confidence (kpt_shape [8, 2]).
    static constexpr int kCorners = 8;
    int kd = 0; // values per corner: 3 (x, y, conf) or 2 (x, y)
    int corners_start = 0;
    int num_classes = 0;
    if (end_to_end) {
        if ((channels - 6) != kCorners * 3) {
            return {};
        }
        kd = 3;
        corners_start = 6;
    } else if (!has_objectness) {
        // 4 box + nc scores + 8 corner values.
        num_classes = channels - 4 - kCorners * 3;
        if (num_classes >= 1) {
            kd = 3;
        } else {
            num_classes = channels - 4 - kCorners * 2;
            if (num_classes >= 1) {
                kd = 2;
            } else {
                return {};
            }
        }
        corners_start = 4 + num_classes;
    } else {
        // 4 box + obj + nc scores + 8 corner values.
        num_classes = channels - 5 - kCorners * 3;
        if (num_classes >= 1) {
            kd = 3;
        } else {
            num_classes = channels - 5 - kCorners * 2;
            if (num_classes >= 1) {
                kd = 2;
            } else {
                return {};
            }
        }
        corners_start = 5 + num_classes;
    }

    std::vector<Monocular3DDetection> detections;
    for (int b = 0; b < batch; ++b) {
        const size_t batch_offset = static_cast<size_t>(b) * batch_stride;
        if (batch_offset + batch_stride > data.size()) {
            break;
        }
        for (int i = 0; i < anchors; ++i) {
            float score = 0.0f;
            int class_id = 0;
            float cx = 0.0f;
            float cy = 0.0f;
            float w = 0.0f;
            float h = 0.0f;

            if (end_to_end) {
                const size_t row = batch_offset + static_cast<size_t>(i) * static_cast<size_t>(channels);
                const float x1 = tensorElementToFloat(data[row + 0]);
                const float y1 = tensorElementToFloat(data[row + 1]);
                const float x2 = tensorElementToFloat(data[row + 2]);
                const float y2 = tensorElementToFloat(data[row + 3]);
                score = tensorElementToFloat(data[row + 4]);
                if (score < confidence_threshold_) {
                    continue;
                }
                class_id = tensorElementToInt(data[row + 5]);
                cx = (x1 + x2) * 0.5f;
                cy = (y1 + y2) * 0.5f;
                w = x2 - x1;
                h = y2 - y1;
            } else if (!has_objectness) {
                float best = 0.0f;
                for (int c = 0; c < num_classes; ++c) {
                    const float value =
                        tensorElementToFloat(data[batch_offset + static_cast<size_t>((4 + c) * anchors + i)]);
                    if (c == 0 || value > best) {
                        best = value;
                        class_id = c;
                    }
                }
                score = best;
                if (score < confidence_threshold_) {
                    continue;
                }
                cx = tensorElementToFloat(data[batch_offset + static_cast<size_t>(0 * anchors + i)]);
                cy = tensorElementToFloat(data[batch_offset + static_cast<size_t>(1 * anchors + i)]);
                w = tensorElementToFloat(data[batch_offset + static_cast<size_t>(2 * anchors + i)]);
                h = tensorElementToFloat(data[batch_offset + static_cast<size_t>(3 * anchors + i)]);
            } else {
                const size_t row = batch_offset + static_cast<size_t>(i) * static_cast<size_t>(channels);
                const float objectness = tensorElementToFloat(data[row + 4]);
                if (objectness < confidence_threshold_) {
                    continue;
                }
                float best = 0.0f;
                for (int c = 0; c < num_classes; ++c) {
                    const float value = tensorElementToFloat(data[row + static_cast<size_t>(5 + c)]);
                    if (c == 0 || value > best) {
                        best = value;
                        class_id = c;
                    }
                }
                score = objectness * best;
                if (score < confidence_threshold_) {
                    continue;
                }
                cx = tensorElementToFloat(data[row + 0]);
                cy = tensorElementToFloat(data[row + 1]);
                w = tensorElementToFloat(data[row + 2]);
                h = tensorElementToFloat(data[row + 3]);
            }

            Monocular3DDetection detection(scaleBoxToOriginal(cx, cy, w, h, original_size), score, class_id);
            float corner_conf_sum = 0.0f;
            for (int k = 0; k < kCorners; ++k) {
                float kx = 0.0f;
                float ky = 0.0f;
                // v1 exports carry no per-corner confidence: fall back to the detection score.
                float kconf = score;
                if (!has_objectness) {
                    kx = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>((corners_start + k * kd + 0) * anchors + i)]);
                    ky = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>((corners_start + k * kd + 1) * anchors + i)]);
                    if (kd == 3) {
                        kconf = tensorElementToFloat(
                            data[batch_offset + static_cast<size_t>((corners_start + k * kd + 2) * anchors + i)]);
                    }
                } else {
                    const size_t row = batch_offset + static_cast<size_t>(i) * static_cast<size_t>(channels);
                    kx = tensorElementToFloat(data[row + static_cast<size_t>(corners_start + k * kd + 0)]);
                    ky = tensorElementToFloat(data[row + static_cast<size_t>(corners_start + k * kd + 1)]);
                    if (kd == 3) {
                        kconf = tensorElementToFloat(data[row + static_cast<size_t>(corners_start + k * kd + 2)]);
                    }
                }
                const auto scaled = scalePointToOriginal(kx, ky, original_size);
                // Opt-in remap for paper-text-conformant top-first checkpoints.
                const size_t slot = static_cast<size_t>(top_first_ ? (k + kCorners / 2) % kCorners : k);
                detection.projected_corners[slot].x = scaled.first;
                detection.projected_corners[slot].y = scaled.second;
                detection.projected_corners[slot].confidence = kconf;
                corner_conf_sum += kconf;
            }
            detection.geometry_confidence = corner_conf_sum / static_cast<float>(kCorners);
            detections.push_back(detection);
        }
    }

    applyNMS(detections);
    return detections;
}

} // namespace neuriplo_tasks
