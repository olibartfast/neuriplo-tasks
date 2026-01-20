#include "vision-core/pose_estimation/vit_pose_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace vision_core {

ViTPosePostprocessor::ViTPosePostprocessor() {}

std::vector<PoseEstimation> ViTPosePostprocessor::postprocess(
        const std::vector<Tensor>& tensors,
        const cv::Size& original_size,
        const cv::Size& input_size) {
    
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
                    const auto& val_variant = data[offset + y * heatmap_w + x];
                    float val = 0.0f;
                    if (std::holds_alternative<float>(val_variant)) {
                        val = std::get<float>(val_variant);
                    } else if (std::holds_alternative<int32_t>(val_variant)) {
                        val = static_cast<float>(std::get<int32_t>(val_variant));
                    } else if (std::holds_alternative<int64_t>(val_variant)) {
                        val = static_cast<float>(std::get<int64_t>(val_variant));
                    }
                    
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
            
            kp.x = static_cast<float>(max_x) * original_size.width / heatmap_w;
            kp.y = static_cast<float>(max_y) * original_size.height / heatmap_h;
            kp.confidence = max_val;
            
            pose.keypoints.push_back(kp);
            total_score += max_val;
        }
        
        if (num_joints > 0) {
            pose.score = total_score / num_joints;
        }
        results.push_back(pose);
    }

    return results;
}

} // namespace vision_core
