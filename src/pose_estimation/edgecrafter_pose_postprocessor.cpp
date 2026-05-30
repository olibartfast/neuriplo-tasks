#include "vision-core/pose_estimation/edgecrafter_pose_postprocessor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vision_core {

EdgeCrafterPosePostprocessor::EdgeCrafterPosePostprocessor(float confidence_threshold, float keypoint_threshold,
                                                           const std::vector<std::string>& output_names)
    : confidence_threshold_(confidence_threshold), keypoint_threshold_(keypoint_threshold) {
    findOutputIndices(output_names);
}

void EdgeCrafterPosePostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    if (output_names.empty()) {
        return;
    }

    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& name = output_names[i];
        if (name == "scores") {
            scores_idx_ = static_cast<int>(i);
        } else if (name == "keypoints") {
            keypoints_idx_ = static_cast<int>(i);
        } else if (name == "labels") {
            labels_idx_ = static_cast<int>(i);
        }
    }
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

    int num_dets = static_cast<int>(scores_tensor.shape[1]);
    int num_kpts = static_cast<int>(keypoints_tensor.shape[2]);
    int kpt_dim = static_cast<int>(keypoints_tensor.shape[3]);

    if (kpt_dim != 2 && kpt_dim != 3) {
        return {};
    }

    std::vector<PoseEstimation> poses;
    poses.reserve(static_cast<size_t>(num_dets));

    static constexpr int kLabelOffset = -1;

    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(scores_tensor.data[static_cast<size_t>(i)]);

        if (score < confidence_threshold_) {
            continue;
        }

        int class_id = getTensorInt(labels_tensor.data[static_cast<size_t>(i)]) + kLabelOffset;
        if (class_id < 0) {
            continue;
        }

        PoseEstimation pose;
        pose.score = score;
        pose.bbox = {};

        pose.keypoints.reserve(static_cast<size_t>(num_kpts));
        for (int k = 0; k < num_kpts; ++k) {
            float kx =
                getTensorFloat(keypoints_tensor.data[static_cast<size_t>(i * num_kpts * kpt_dim + k * kpt_dim + 0)]);
            float ky =
                getTensorFloat(keypoints_tensor.data[static_cast<size_t>(i * num_kpts * kpt_dim + k * kpt_dim + 1)]);
            float kconf =
                kpt_dim == 3 ? getTensorFloat(
                                   keypoints_tensor.data[static_cast<size_t>(i * num_kpts * kpt_dim + k * kpt_dim + 2)])
                             : 1.0F;

            Keypoint kp;
            kp.x = kx;
            kp.y = ky;
            kp.confidence = kconf;
            pose.keypoints.push_back(kp);
        }

        deriveBboxFromKeypoints(pose);
        poses.push_back(std::move(pose));
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
        pose.bbox = cv::Rect(static_cast<int>(x_min), static_cast<int>(y_min), static_cast<int>(x_max - x_min),
                             static_cast<int>(y_max - y_min));
    }
}

float EdgeCrafterPosePostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

int EdgeCrafterPosePostprocessor::getTensorInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int { return static_cast<int>(value); }, element);
}

} // namespace vision_core
