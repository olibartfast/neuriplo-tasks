#include "neuriplo/tasks/pose_estimation/yolo_pose_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

namespace neuriplo_tasks {

YoloPosePostprocessor::YoloPosePostprocessor(const vision::Size& input_size, float confidence_threshold,
                                             float nms_threshold)
    : input_size_(input_size), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold) {}

vision::Rect YoloPosePostprocessor::scaleBoxToOriginal(float cx, float cy, float w, float h,
                                                       const vision::Size& frame_size) const {
    float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);

    int x, y, width, height;
    if (r_h > r_w) {
        float pad_h = (static_cast<float>(input_size_.height) - r_w * static_cast<float>(frame_size.height)) / 2.0f;
        x = static_cast<int>((cx - w / 2.0f) / r_w);
        y = static_cast<int>((cy - h / 2.0f - pad_h) / r_w);
        width = static_cast<int>(w / r_w);
        height = static_cast<int>(h / r_w);
    } else {
        float pad_w = (static_cast<float>(input_size_.width) - r_h * static_cast<float>(frame_size.width)) / 2.0f;
        x = static_cast<int>((cx - w / 2.0f - pad_w) / r_h);
        y = static_cast<int>((cy - h / 2.0f) / r_h);
        width = static_cast<int>(w / r_h);
        height = static_cast<int>(h / r_h);
    }
    return vision::Rect(x, y, width, height);
}

vision::Point2f YoloPosePostprocessor::scaleKptToOriginal(float kx, float ky, const vision::Size& frame_size) const {
    float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);

    float x, y;
    if (r_h > r_w) {
        float pad_h = (static_cast<float>(input_size_.height) - r_w * static_cast<float>(frame_size.height)) / 2.0f;
        x = kx / r_w;
        y = (ky - pad_h) / r_w;
    } else {
        float pad_w = (static_cast<float>(input_size_.width) - r_h * static_cast<float>(frame_size.width)) / 2.0f;
        x = (kx - pad_w) / r_h;
        y = ky / r_h;
    }
    return vision::Point2f(x, y);
}

void YoloPosePostprocessor::applyNMS(std::vector<PoseEstimation>& poses) const {
    if (poses.empty())
        return;

    std::vector<image_ops::DetectionBox> det_boxes;
    det_boxes.reserve(poses.size());
    for (const auto& p : poses) {
        det_boxes.push_back({p.bbox, p.score, 0});
    }

    auto keep_indices = image_ops::nms(det_boxes, nms_threshold_);

    std::vector<PoseEstimation> filtered;
    filtered.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        filtered.push_back(poses[static_cast<size_t>(idx)]);
    }
    poses = std::move(filtered);
}

std::vector<PoseEstimation> YoloPosePostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                               const vision::Size& original_size,
                                                               const vision::Size& /*input_size*/) {
    if (tensors.empty())
        return {};

    const auto& tensor = tensors[0];
    const auto& shape = tensor.shape;
    const auto& data = tensor.data;

    // Require at least [batch, dim1, dim2]
    if (shape.size() < 3)
        return {};

    const int batch = static_cast<int>(shape[0]);
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);

    // YOLOv5 (with objectness): [batch, anchors, channels] → dim2 < dim1
    // YOLOv8+  (no objectness): [batch, channels, anchors] → dim1 < dim2
    bool has_objectness = (dim2 < dim1);

    int channels = has_objectness ? dim2 : dim1;
    int anchors = has_objectness ? dim1 : dim2;
    const size_t batch_stride = static_cast<size_t>(channels) * static_cast<size_t>(anchors);

    // Determine keypoint layout:
    //   Format A: 4 bbox + 1 conf + 3*kpts          → kpts_start = 5
    //   Format B: 4 bbox + 1 obj + 1 cls + 3*kpts   → kpts_start = 6  (YOLOv5 with class score)
    int kpts_start = 5;
    int num_kpts = 0;

    if (channels > 5 && (channels - 5) % 3 == 0) {
        kpts_start = 5;
        num_kpts = (channels - 5) / 3;
    } else if (channels > 6 && (channels - 6) % 3 == 0) {
        kpts_start = 6;
        num_kpts = (channels - 6) / 3;
    } else {
        return {}; // Unrecognized channel layout
    }

    std::vector<PoseEstimation> poses;

    for (int b = 0; b < batch; ++b) {
        const size_t batch_offset = static_cast<size_t>(b) * batch_stride;

        for (int i = 0; i < anchors; ++i) {
            float conf = 0.0f;
            float cx, cy, w, h;

            if (has_objectness) {
                float obj = tensorElementToFloat(data[batch_offset + static_cast<size_t>(i * channels + 4)]);
                if (obj < confidence_threshold_)
                    continue;

                if (kpts_start == 6) {
                    float cls = tensorElementToFloat(data[batch_offset + static_cast<size_t>(i * channels + 5)]);
                    conf = obj * cls;
                } else {
                    conf = obj;
                }

                if (conf < confidence_threshold_)
                    continue;

                cx = tensorElementToFloat(data[batch_offset + static_cast<size_t>(i * channels + 0)]);
                cy = tensorElementToFloat(data[batch_offset + static_cast<size_t>(i * channels + 1)]);
                w = tensorElementToFloat(data[batch_offset + static_cast<size_t>(i * channels + 2)]);
                h = tensorElementToFloat(data[batch_offset + static_cast<size_t>(i * channels + 3)]);
            } else {
                conf = tensorElementToFloat(data[batch_offset + static_cast<size_t>(4 * anchors + i)]);
                if (conf < confidence_threshold_)
                    continue;

                cx = tensorElementToFloat(data[batch_offset + static_cast<size_t>(0 * anchors + i)]);
                cy = tensorElementToFloat(data[batch_offset + static_cast<size_t>(1 * anchors + i)]);
                w = tensorElementToFloat(data[batch_offset + static_cast<size_t>(2 * anchors + i)]);
                h = tensorElementToFloat(data[batch_offset + static_cast<size_t>(3 * anchors + i)]);
            }

            PoseEstimation pose;
            pose.score = conf;
            pose.bbox = scaleBoxToOriginal(cx, cy, w, h, original_size);

            pose.keypoints.reserve(static_cast<size_t>(num_kpts));
            for (int j = 0; j < num_kpts; ++j) {
                float kx, ky, kconf;
                if (has_objectness) {
                    kx = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>(i * channels + kpts_start + j * 3 + 0)]);
                    ky = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>(i * channels + kpts_start + j * 3 + 1)]);
                    kconf = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>(i * channels + kpts_start + j * 3 + 2)]);
                } else {
                    kx = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>((kpts_start + j * 3 + 0) * anchors + i)]);
                    ky = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>((kpts_start + j * 3 + 1) * anchors + i)]);
                    kconf = tensorElementToFloat(
                        data[batch_offset + static_cast<size_t>((kpts_start + j * 3 + 2) * anchors + i)]);
                }

                vision::Point2f scaled = scaleKptToOriginal(kx, ky, original_size);
                pose.keypoints.push_back({scaled.x, scaled.y, kconf});
            }

            poses.push_back(std::move(pose));
        }
    }

    applyNMS(poses);
    return poses;
}

} // namespace neuriplo_tasks
