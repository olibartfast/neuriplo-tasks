#include "vision-core/pose_estimation/pose_estimation_task.hpp"
#include "vision-core/pose_estimation/vit_pose_postprocessor.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace vision_core;

TEST(ViTPosePostprocessorTest, BasicPostprocess) {
    ViTPosePostprocessor processor;

    int num_joints = 17;
    int heatmap_h = 64;
    int heatmap_w = 48;

    std::vector<TensorElement> heatmap_data(num_joints * heatmap_h * heatmap_w, 0.0f);

    // Set a peak for the first joint (Nose) at (24, 32) in heatmap
    // offset = joint_idx * h * w + y * w + x
    int x_peak = 24;
    int y_peak = 32;
    heatmap_data[0 * heatmap_h * heatmap_w + y_peak * heatmap_w + x_peak] = 1.0f;

    std::vector<Tensor> tensors = {Tensor(heatmap_data, {1, num_joints, heatmap_h, heatmap_w})};

    cv::Size original_size(1920, 1080);
    cv::Size input_size(192, 256);

    auto results = processor.postprocess(tensors, original_size, input_size);

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].keypoints.size(), num_joints);

    // Check first keypoint mapping
    // kp.x = max_x * original_width / heatmap_w = 24 * 1920 / 48 = 24 * 40 = 960
    // kp.y = max_y * original_height / heatmap_h = 32 * 1080 / 64 = 32 * 16.875 = 540
    EXPECT_NEAR(results[0].keypoints[0].x, 960.0f, 0.1f);
    EXPECT_NEAR(results[0].keypoints[0].y, 540.0f, 0.1f);
    EXPECT_NEAR(results[0].keypoints[0].confidence, 1.0f, 0.01f);
}

TEST(PoseEstimationTaskTest, TaskType) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 256, 192}};
    info.input_formats = {"FORMAT_NCHW"};

    PoseEstimationTask task(info, "vitpose");
    EXPECT_EQ(task.getTaskType(), TaskType::PoseEstimation);
}
