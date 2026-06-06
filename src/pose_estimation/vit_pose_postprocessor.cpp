#include "neuriplo/tasks/pose_estimation/vit_pose_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace neuriplo_tasks {

ViTPosePostprocessor::ViTPosePostprocessor() {}

std::vector<PoseEstimation> ViTPosePostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                              const cv::Size& original_size,
                                                              const cv::Size& /*input_size*/) {

    if (tensors.empty()) {
        return {};
    }

    const auto& output_tensor = tensors[0];
    const auto& shape = output_tensor.shape;
    const auto& data = output_tensor.data;

    // Expected shape: [Batch, Joints, H, W]
    if (shape.size() != 4) {
        // Logic to handle unexpected shape? For now, return empty or try to adapt.
        // If it's flattened, we might need more info. But Tensor usually has shape.
        return {};
    }

    int batch_size = static_cast<int>(shape[0]);
    int num_joints = static_cast<int>(shape[1]);
    int heatmap_h = static_cast<int>(shape[2]);
    int heatmap_w = static_cast<int>(shape[3]);

    std::vector<PoseEstimation> results;

    for (int b = 0; b < batch_size; ++b) {
        PoseEstimation pose;
        float total_score = 0.0f;

        for (int i = 0; i < num_joints; ++i) {
            float max_val = -1e9;
            int max_x = 0;
            int max_y = 0;

            int offset = b * num_joints * heatmap_h * heatmap_w + i * heatmap_h * heatmap_w;

            for (int y = 0; y < heatmap_h; ++y) {
                for (int x = 0; x < heatmap_w; ++x) {
                    const float val = tensorElementToFloat(data[static_cast<size_t>(offset + y * heatmap_w + x)]);

                    if (val > max_val) {
                        max_val = val;
                        max_x = x;
                        max_y = y;
                    }
                }
            }

            Keypoint kp;
            // Map to original image coordinates
            // Assuming the whole image was resized to input_size, and then fed to network.
            // Heatmap is a downsampled version of input.
            // Mapping: heatmap_coord / heatmap_size * original_size

            kp.x = static_cast<float>(max_x) * static_cast<float>(original_size.width) / static_cast<float>(heatmap_w);
            kp.y = static_cast<float>(max_y) * static_cast<float>(original_size.height) / static_cast<float>(heatmap_h);
            kp.confidence = max_val;

            pose.keypoints.push_back(kp);
            total_score += max_val;
        }

        if (num_joints > 0) {
            pose.score = total_score / static_cast<float>(num_joints);
        }
        results.push_back(pose);
    }

    return results;
}

} // namespace neuriplo_tasks
