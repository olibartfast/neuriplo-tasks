#include "neuriplo/tasks/pose_estimation/edgecrafter_pose_postprocessor.hpp"

#include "neuriplo/tasks/core/output_name_utils.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace neuriplo_tasks {

EdgeCrafterPosePostprocessor::EdgeCrafterPosePostprocessor(float confidence_threshold, float keypoint_threshold,
                                                           const std::vector<std::string>& output_names)
    : confidence_threshold_(confidence_threshold), keypoint_threshold_(keypoint_threshold) {
    findOutputIndices(output_names);
}

void EdgeCrafterPosePostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    scores_idx_ = findOutputIndexByName(output_names, "scores", scores_idx_);
    keypoints_idx_ = findOutputIndexByName(output_names, "keypoints", keypoints_idx_);
    labels_idx_ = findOutputIndexByName(output_names, "labels", labels_idx_);
}

std::vector<PoseEstimation> EdgeCrafterPosePostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                      const cv::Size& /*original_size*/,
                                                                      const cv::Size& /*input_size*/) {

    if (tensors.size() < 3) {
        throw std::runtime_error("EdgeCrafter pose requires 3 output tensors (labels, scores, keypoints)");
    }

    const auto& scores_tensor = tensors[static_cast<size_t>(scores_idx_)];
    const auto& keypoints_tensor = tensors[static_cast<size_t>(keypoints_idx_)];
    const auto& labels_tensor = tensors[static_cast<size_t>(labels_idx_)];

    if (scores_tensor.shape.size() < 2 || keypoints_tensor.shape.size() < 4 || labels_tensor.shape.size() < 2) {
        return {};
    }

    const int batch = static_cast<int>(scores_tensor.shape[0]);
    const int num_dets = static_cast<int>(scores_tensor.shape[1]);
    const int num_kpts = static_cast<int>(keypoints_tensor.shape[2]);
    const int kpt_dim = static_cast<int>(keypoints_tensor.shape[3]);

    if (kpt_dim != 2 && kpt_dim != 3) {
        return {};
    }

    const size_t batch_scores_stride = static_cast<size_t>(num_dets);
    const size_t batch_keypoints_stride =
        static_cast<size_t>(num_dets) * static_cast<size_t>(num_kpts) * static_cast<size_t>(kpt_dim);
    const size_t batch_labels_stride = static_cast<size_t>(num_dets);

    std::vector<PoseEstimation> poses;
    poses.reserve(static_cast<size_t>(batch * num_dets));

    static constexpr int kLabelOffset = -1;

    for (int b = 0; b < batch; ++b) {
        const size_t scores_batch_offset = static_cast<size_t>(b) * batch_scores_stride;
        const size_t keypoints_batch_offset = static_cast<size_t>(b) * batch_keypoints_stride;
        const size_t labels_batch_offset = static_cast<size_t>(b) * batch_labels_stride;

        for (int i = 0; i < num_dets; ++i) {
            float score = tensorElementToFloat(scores_tensor.data[scores_batch_offset + static_cast<size_t>(i)]);

            if (score < confidence_threshold_) {
                continue;
            }

            int class_id =
                tensorElementToInt(labels_tensor.data[labels_batch_offset + static_cast<size_t>(i)]) + kLabelOffset;
            if (class_id < 0) {
                continue;
            }

            PoseEstimation pose;
            pose.score = score;
            pose.bbox = {};

            pose.keypoints.reserve(static_cast<size_t>(num_kpts));
            const size_t detection_offset = keypoints_batch_offset + static_cast<size_t>(i) *
                                                                         static_cast<size_t>(num_kpts) *
                                                                         static_cast<size_t>(kpt_dim);
            for (int k = 0; k < num_kpts; ++k) {
                const size_t keypoint_offset = detection_offset + static_cast<size_t>(k) * static_cast<size_t>(kpt_dim);
                float kx = tensorElementToFloat(keypoints_tensor.data[keypoint_offset + 0U]);
                float ky = tensorElementToFloat(keypoints_tensor.data[keypoint_offset + 1U]);
                float kconf = kpt_dim == 3 ? tensorElementToFloat(keypoints_tensor.data[keypoint_offset + 2U]) : 1.0F;

                Keypoint kp;
                kp.x = kx;
                kp.y = ky;
                kp.confidence = kconf;
                pose.keypoints.push_back(kp);
            }

            deriveBboxFromKeypoints(pose);
            poses.push_back(std::move(pose));
        }
    }

    return poses;
}

void EdgeCrafterPosePostprocessor::deriveBboxFromKeypoints(PoseEstimation& pose) const {
    float x_min = std::numeric_limits<float>::max();
    float y_min = std::numeric_limits<float>::max();
    float x_max = 0.0F;
    float y_max = 0.0F;
    bool any = false;

    for (const auto& kp : pose.keypoints) {
        if (kp.confidence > keypoint_threshold_) {
            x_min = std::min(x_min, kp.x);
            y_min = std::min(y_min, kp.y);
            x_max = std::max(x_max, kp.x);
            y_max = std::max(y_max, kp.y);
            any = true;
        }
    }

    if (any) {
        pose.bbox = BoundingBox(static_cast<int>(x_min), static_cast<int>(y_min), static_cast<int>(x_max - x_min),
                                static_cast<int>(y_max - y_min));
    }
}

} // namespace neuriplo_tasks
