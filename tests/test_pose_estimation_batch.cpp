#include "neuriplo/tasks/core/batch_postprocess.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/pose_estimation/vit_pose_postprocessor.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

namespace {

ModelInfo vitPoseModelInfo() {
    ModelInfo info;
    info.input_shapes = {{2, 3, 256, 192}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.batch_size_ = 2;
    info.max_batch_size_ = 2;
    return info;
}

Tensor makeHeatmapTensor(int batch, int joints, int heatmap_h, int heatmap_w, int peak_batch) {
    std::vector<TensorElement> data(static_cast<size_t>(batch * joints * heatmap_h * heatmap_w), 0.0f);
    const int peak_x = 10;
    const int peak_y = 12;
    const size_t offset = static_cast<size_t>(peak_batch) * static_cast<size_t>(joints * heatmap_h * heatmap_w);
    data[offset + static_cast<size_t>(peak_y * heatmap_w + peak_x)] = 1.0f;
    return Tensor(std::move(data), {batch, joints, heatmap_h, heatmap_w});
}

} // namespace

TEST(PoseEstimationBatchTest, ViTPosePostprocessorSplitsBatch) {
    ViTPosePostprocessor processor;

    const int joints = 17;
    const int heatmap_h = 32;
    const int heatmap_w = 24;
    const std::vector<Tensor> tensors = {makeHeatmapTensor(2, joints, heatmap_h, heatmap_w, 1)};

    const neuriplo_tasks::Size frame_size(192, 256);
    const neuriplo_tasks::Size input_size(192, 256);
    const auto poses = processor.postprocess(tensors, frame_size, input_size);

    ASSERT_EQ(poses.size(), 2u);
    EXPECT_NEAR(poses[1].keypoints[0].x, 10.0f * frame_size.width / heatmap_w, 0.1f);
    EXPECT_NEAR(poses[1].keypoints[0].y, 12.0f * frame_size.height / heatmap_h, 0.1f);
}

TEST(PoseEstimationBatchTest, TaskPostprocessBatchSizeTwo) {
    auto task = TaskFactory::createTaskInstance("vitpose", vitPoseModelInfo());
    ASSERT_NE(task, nullptr);

    const int joints = 17;
    const std::vector<Tensor> tensors = {makeHeatmapTensor(2, joints, 32, 24, 0)};

    const auto results = task->postprocess(neuriplo_tasks::Size(192, 256), tensors);

    ASSERT_EQ(results.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<PoseEstimation>(results[0]));
}

TEST(PoseEstimationBatchTest, BatchPostprocessMatchesBatchSize) {
    auto task = TaskFactory::createTaskInstance("vitpose", vitPoseModelInfo());
    ASSERT_NE(task, nullptr);

    const std::vector<Tensor> tensors = {makeHeatmapTensor(2, 17, 32, 24, 0)};
    const auto post = batchPostprocess(*task, neuriplo_tasks::Size(192, 256), tensors, 2);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(post));
}
