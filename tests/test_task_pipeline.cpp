#include "neuriplo/tasks/core/opencv_interop.hpp"
#include "neuriplo/tasks/core/task_pipeline.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

using namespace neuriplo_tasks;

namespace {

Detection makeDetection() { return Detection(BoundingBox(10, 20, 30, 40), 0.9F, 1); }

std::vector<Result> detectionToPose(const std::vector<Result>& inputs) {
    std::vector<Result> outputs;
    outputs.reserve(inputs.size());

    for (const auto& input : inputs) {
        if (!std::holds_alternative<Detection>(input)) {
            continue;
        }

        const auto& detection = std::get<Detection>(input);
        PoseEstimation pose;
        pose.bbox = detection.bbox;
        pose.score = detection.class_confidence;
        pose.keypoints.push_back(
            {static_cast<float>(detection.bbox.x), static_cast<float>(detection.bbox.y), detection.class_confidence});
        outputs.emplace_back(std::move(pose));
    }

    return outputs;
}

std::vector<Result> detectionToSegmentation(const std::vector<Result>& inputs) {
    std::vector<Result> outputs;
    outputs.reserve(inputs.size());

    for (const auto& input : inputs) {
        if (!std::holds_alternative<Detection>(input)) {
            continue;
        }

        const auto& detection = std::get<Detection>(input);
        InstanceSegmentation segmentation(detection.bbox, detection.class_confidence,
                                          static_cast<int>(detection.class_id));
        segmentation.mask = fromCvMat(cv::Mat::ones(detection.bbox.height, detection.bbox.width, CV_8UC1));
        outputs.emplace_back(std::move(segmentation));
    }

    return outputs;
}

} // namespace

TEST(TaskPipelineTest, EmptySequentialPipelineReturnsInputs) {
    SequentialTaskPipeline pipeline;
    const std::vector<Result> inputs = {makeDetection()};

    const auto outputs = pipeline.run(inputs);

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<Detection>(outputs[0]));
    EXPECT_EQ(std::get<Detection>(outputs[0]).bbox, BoundingBox(10, 20, 30, 40));
}

TEST(TaskPipelineTest, RejectsEmptyStage) {
    SequentialTaskPipeline pipeline;

    EXPECT_THROW(pipeline.addStage({}), std::invalid_argument);
}

TEST(TaskPipelineTest, RunsDetectionToPoseStagesInOrder) {
    std::vector<int> order;
    SequentialTaskPipeline pipeline;
    pipeline.addStage([&order](const std::vector<Result>& inputs) {
        order.push_back(1);
        return detectionToPose(inputs);
    });
    pipeline.addStage([&order](const std::vector<Result>& inputs) {
        order.push_back(2);
        std::vector<Result> outputs;
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            auto pose = std::get<PoseEstimation>(input);
            pose.score += 0.1F;
            outputs.emplace_back(std::move(pose));
        }
        return outputs;
    });

    const auto outputs = pipeline.run({makeDetection()});

    ASSERT_EQ(order, std::vector<int>({1, 2}));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<PoseEstimation>(outputs[0]));
    const auto& pose = std::get<PoseEstimation>(outputs[0]);
    EXPECT_EQ(pose.bbox, BoundingBox(10, 20, 30, 40));
    EXPECT_NEAR(pose.score, 1.0F, 0.001F);
    ASSERT_EQ(pose.keypoints.size(), 1u);
}

TEST(TaskPipelineTest, SupportsDetectionToSegmentationPipeline) {
    SequentialTaskPipeline pipeline({detectionToSegmentation});

    const auto outputs = pipeline.run({makeDetection()});

    ASSERT_EQ(pipeline.stageCount(), 1u);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<InstanceSegmentation>(outputs[0]));
    const auto& segmentation = std::get<InstanceSegmentation>(outputs[0]);
    EXPECT_EQ(segmentation.bbox, BoundingBox(10, 20, 30, 40));
    EXPECT_EQ(segmentation.mask.rows(), 40);
    EXPECT_EQ(segmentation.mask.cols(), 30);
}
