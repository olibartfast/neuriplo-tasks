#include "neuriplo/tasks/pose_estimation/rfdetr_pose_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace neuriplo_tasks {

RfDetrPosePostprocessor::RfDetrPosePostprocessor(const cv::Size& input_size, float confidence_threshold,
                                                 float keypoint_uncertainty_alpha,
                                                 const std::vector<int>& keypoint_counts)
    : input_size_(input_size), confidence_threshold_(confidence_threshold),
      keypoint_uncertainty_alpha_(keypoint_uncertainty_alpha), keypoint_counts_(keypoint_counts) {}

std::vector<PoseEstimation> RfDetrPosePostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                 const cv::Size& original_size,
                                                                 const cv::Size& /*input_size*/) {
    if (tensors.size() < 3) {
        throw std::runtime_error("RF-DETR pose requires 3 output tensors (dets, labels, keypoints)");
    }

    const auto& dets_tensor = tensors[0];
    const auto& labels_tensor = tensors[1];
    const auto& keypoints_tensor = tensors[2];

    const auto& dets_shape = dets_tensor.shape;
    const auto& labels_shape = labels_tensor.shape;
    const auto& keypoints_shape = keypoints_tensor.shape;

    if (dets_shape.size() < 3 || labels_shape.size() < 3 || keypoints_shape.size() < 4) {
        return {};
    }

    int num_dets = static_cast<int>(dets_shape[1]);
    int num_classes = static_cast<int>(labels_shape[2]);
    int keypoint_dim2 = static_cast<int>(keypoints_shape[2]);
    int keypoint_channels = static_cast<int>(keypoints_shape[3]);

    if (keypoint_channels != 8) {
        return {};
    }

    int K_max = num_classes > 0 ? keypoint_dim2 / num_classes : 0;

    const auto& dets_data = dets_tensor.data;
    const auto& labels_data = labels_tensor.data;
    const auto& keypoints_data = keypoints_tensor.data;

    float scale_w = static_cast<float>(original_size.width) / static_cast<float>(input_size_.width);
    float scale_h = static_cast<float>(original_size.height) / static_cast<float>(input_size_.height);

    std::vector<PoseEstimation> results;

    for (int i = 0; i < num_dets; ++i) {
        float max_score = 0.0f;
        int max_class_idx = -1;

        for (int c = 0; c < num_classes; ++c) {
            float logit = tensorElementToFloat(labels_data[static_cast<size_t>(i * num_classes + c)]);
            float score = 1.0f / (1.0f + std::exp(-logit));
            if (score > max_score) {
                max_score = score;
                max_class_idx = c;
            }
        }

        if (max_score < confidence_threshold_) {
            continue;
        }

        max_class_idx -= 1;
        if (max_class_idx < 0) {
            continue;
        }

        float cx =
            tensorElementToFloat(dets_data[static_cast<size_t>(i * 4 + 0)]) * static_cast<float>(input_size_.width);
        float cy =
            tensorElementToFloat(dets_data[static_cast<size_t>(i * 4 + 1)]) * static_cast<float>(input_size_.height);
        float w =
            tensorElementToFloat(dets_data[static_cast<size_t>(i * 4 + 2)]) * static_cast<float>(input_size_.width);
        float h =
            tensorElementToFloat(dets_data[static_cast<size_t>(i * 4 + 3)]) * static_cast<float>(input_size_.height);

        float x = (cx - w / 2.0f) * scale_w;
        float y = (cy - h / 2.0f) * scale_h;
        float width = w * scale_w;
        float height = h * scale_h;

        int num_kpts = 0;
        if (max_class_idx < static_cast<int>(keypoint_counts_.size())) {
            num_kpts = keypoint_counts_[static_cast<size_t>(max_class_idx)];
        }

        std::vector<Keypoint> keypoints;
        keypoints.reserve(static_cast<size_t>(num_kpts));

        float trace_sum = 0.0f;

        for (int k = 0; k < num_kpts; ++k) {
            size_t base = static_cast<size_t>(i * keypoint_dim2 + max_class_idx * K_max + k) *
                          static_cast<size_t>(keypoint_channels);

            float kx = tensorElementToFloat(keypoints_data[base + 0]) * static_cast<float>(original_size.width);
            float ky = tensorElementToFloat(keypoints_data[base + 1]) * static_cast<float>(original_size.height);

            float findability = 1.0f / (1.0f + std::exp(-tensorElementToFloat(keypoints_data[base + 2])));
            float visibility = 1.0f / (1.0f + std::exp(-tensorElementToFloat(keypoints_data[base + 3])));

            float L11 = std::exp(tensorElementToFloat(keypoints_data[base + 4]));
            float L21 = tensorElementToFloat(keypoints_data[base + 5]);
            float L22 = std::exp(tensorElementToFloat(keypoints_data[base + 6]));

            float p00 = L11 * L11;
            float p01 = L11 * L21;
            float p10 = p01;
            float p11 = L21 * L21 + L22 * L22;

            float det = p00 * p11 - p01 * p10;
            float cov00, cov01, cov10, cov11;
            if (std::abs(det) > 1e-12f) {
                cov00 = p11 / det;
                cov01 = -p01 / det;
                cov10 = -p10 / det;
                cov11 = p00 / det;
            } else {
                cov00 = 1.0f;
                cov01 = 0.0f;
                cov10 = 0.0f;
                cov11 = 1.0f;
            }

            trace_sum += cov00 + cov11;

            Keypoint kp;
            kp.x = kx;
            kp.y = ky;
            kp.confidence = findability;
            kp.visibility = visibility;
            kp.covariance = {cov00, cov01, cov10, cov11};
            keypoints.push_back(kp);
        }

        float final_score = max_score;
        if (num_kpts > 0 && trace_sum > 0.0f) {
            float trace_avg = trace_sum / static_cast<float>(num_kpts);
            float uncertainty_reduction = std::exp(-keypoint_uncertainty_alpha_ * std::log(trace_avg));
            final_score = max_score * uncertainty_reduction;
        }

        PoseEstimation pose;
        pose.bbox =
            BoundingBox(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height));
        pose.keypoints = std::move(keypoints);
        pose.score = final_score;
        results.push_back(std::move(pose));
    }

    return results;
}

} // namespace neuriplo_tasks
