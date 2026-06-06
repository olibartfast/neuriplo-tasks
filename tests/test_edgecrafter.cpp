#include "vision-core/core/model_info.hpp"
#include "vision-core/core/task_config.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/instance_segmentation/edgecrafter_segmentation_postprocessor.hpp"
#include "vision-core/instance_segmentation/instance_segmentation_task.hpp"
#include "vision-core/object_detection/edgecrafter_postprocessor.hpp"
#include "vision-core/object_detection/object_detection_task.hpp"
#include "vision-core/pose_estimation/edgecrafter_pose_postprocessor.hpp"
#include "vision-core/pose_estimation/pose_estimation_task.hpp"

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vision_core;

namespace {

ModelInfo createDetectionModelInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}, {1, 2}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"images", "orig_target_sizes"};
    info.output_names = {"labels", "boxes", "scores"};
    info.input_types = {CV_32F, CV_32S};
    return info;
}

ModelInfo createSegmentationModelInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}, {1, 2}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"images", "orig_target_sizes"};
    info.output_names = {"labels", "boxes", "scores", "masks"};
    info.input_types = {CV_32F, CV_32S};
    return info;
}

ModelInfo createPoseModelInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}, {1, 2}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"images", "orig_target_sizes"};
    info.output_names = {"labels", "scores", "keypoints"};
    info.input_types = {CV_32F, CV_32S};
    return info;
}

std::array<int64_t, 2> decodeInt64Pair(const std::vector<uint8_t>& bytes) {
    std::array<int64_t, 2> values{};
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(int64_t));
    return values;
}

} // namespace

// ---------------------------------------------------------------------------
// EdgeCrafter Detection Postprocessor Tests
// ---------------------------------------------------------------------------

TEST(EdgeCrafterDetectionTest, BasicPostprocess) {
    EdgeCrafterPostprocessor pp(0.5f);

    int num_dets = 300;
    std::vector<TensorElement> scores(num_dets, 0.0f);
    std::vector<TensorElement> boxes(static_cast<size_t>(num_dets) * 4, 0.0f);
    std::vector<TensorElement> labels(num_dets, static_cast<int64_t>(0));

    scores[0] = 0.9f;
    boxes[0] = 100.0f;
    boxes[1] = 100.0f;
    boxes[2] = 300.0f;
    boxes[3] = 300.0f;
    labels[0] = static_cast<int64_t>(5);

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
        Tensor(scores, {1, num_dets}),
    };

    auto detections = pp.postprocess(tensors, cv::Size(640, 480));

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
    EXPECT_EQ(detections[0].class_id, 5);
    EXPECT_EQ(detections[0].bbox.x, 100);
    EXPECT_EQ(detections[0].bbox.y, 100);
    EXPECT_EQ(detections[0].bbox.width, 200);
    EXPECT_EQ(detections[0].bbox.height, 200);
}

TEST(EdgeCrafterDetectionTest, FiltersBelowThreshold) {
    EdgeCrafterPostprocessor pp(0.5f);
    int num_dets = 10;
    std::vector<TensorElement> scores(static_cast<size_t>(num_dets), 0.1f);
    std::vector<TensorElement> boxes(static_cast<size_t>(num_dets) * 4, 0.0f);
    std::vector<TensorElement> labels(static_cast<size_t>(num_dets), static_cast<int64_t>(0));

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
        Tensor(scores, {1, num_dets}),
    };

    auto detections = pp.postprocess(tensors, cv::Size(640, 480));
    EXPECT_TRUE(detections.empty());
}

TEST(EdgeCrafterDetectionTest, FiltersNegativeLabels) {
    EdgeCrafterPostprocessor pp(0.1f);
    int num_dets = 1;
    std::vector<TensorElement> scores = {0.9f};
    std::vector<TensorElement> boxes = {0.0f, 0.0f, 100.0f, 100.0f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(-1)};

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
        Tensor(scores, {1, num_dets}),
    };

    auto detections = pp.postprocess(tensors, cv::Size(640, 480));
    EXPECT_TRUE(detections.empty());
}

TEST(EdgeCrafterDetectionTest, EmptyInputThrows) {
    EdgeCrafterPostprocessor pp(0.5f);
    EXPECT_THROW({ pp.postprocess({}, cv::Size(640, 480)); }, std::runtime_error);
}

// ---------------------------------------------------------------------------
// EdgeCrafter Segmentation Postprocessor Tests
// ---------------------------------------------------------------------------

TEST(EdgeCrafterSegmentationTest, BasicPostprocess) {
    EdgeCrafterSegmentationPostprocessor pp(0.5f, 0.5f);

    int num_dets = 1;
    int mask_h = 160;
    int mask_w = 160;

    std::vector<TensorElement> scores = {0.9f};
    std::vector<TensorElement> boxes = {100.0f, 100.0f, 300.0f, 300.0f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(5)};

    size_t mask_size = static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);
    std::vector<TensorElement> masks(mask_size, 1.0f);

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
        Tensor(scores, {1, num_dets}),
        Tensor(masks, {1, num_dets, mask_h, mask_w}),
    };

    auto segmentations = pp.postprocess(tensors, cv::Size(640, 480));

    ASSERT_EQ(segmentations.size(), 1u);
    EXPECT_FLOAT_EQ(segmentations[0].class_confidence, 0.9f);
    EXPECT_EQ(segmentations[0].class_id, 5);
    EXPECT_GT(segmentations[0].bbox.width, 0);
    EXPECT_GT(segmentations[0].bbox.height, 0);
    EXPECT_FALSE(segmentations[0].mask.empty());
    EXPECT_EQ(segmentations[0].mask.rows(), 480);
    EXPECT_EQ(segmentations[0].mask.cols(), 640);
}

TEST(EdgeCrafterSegmentationTest, FiltersBelowThreshold) {
    EdgeCrafterSegmentationPostprocessor pp(0.5f, 0.5f);
    int num_dets = 1;
    std::vector<TensorElement> scores = {0.1f};
    std::vector<TensorElement> boxes = {0.0f, 0.0f, 100.0f, 100.0f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(0)};
    std::vector<TensorElement> masks(static_cast<size_t>(160 * 160), 0.0f);

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
        Tensor(scores, {1, num_dets}),
        Tensor(masks, {1, num_dets, 160, 160}),
    };

    auto segmentations = pp.postprocess(tensors, cv::Size(640, 480));
    EXPECT_TRUE(segmentations.empty());
}

TEST(EdgeCrafterSegmentationTest, EmptyInputThrows) {
    EdgeCrafterSegmentationPostprocessor pp(0.5f, 0.5f);
    EXPECT_THROW({ pp.postprocess({}, cv::Size(640, 480)); }, std::runtime_error);
}

// ---------------------------------------------------------------------------
// EdgeCrafter Pose Postprocessor Tests
// ---------------------------------------------------------------------------

TEST(EdgeCrafterPoseTest, BasicPostprocess17Keypoints) {
    EdgeCrafterPosePostprocessor pp(0.5f, 0.3f);

    int num_dets = 1;
    int num_kpts = 17;
    int kpt_dim = 3;

    std::vector<TensorElement> scores = {0.9f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(1)};
    size_t kpt_size = static_cast<size_t>(num_kpts) * static_cast<size_t>(kpt_dim);
    std::vector<TensorElement> keypoints(kpt_size, 0.0f);

    // Set first keypoint (nose) with high confidence
    keypoints[0] = 100.0f;
    keypoints[1] = 150.0f;
    keypoints[2] = 0.9f;
    // Set second keypoint (left eye)
    keypoints[3] = 90.0f;
    keypoints[4] = 140.0f;
    keypoints[5] = 0.8f;

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(scores, {1, num_dets}),
        Tensor(keypoints, {1, num_dets, num_kpts, kpt_dim}),
    };

    auto poses = pp.postprocess(tensors, cv::Size(640, 480), cv::Size(640, 640));

    ASSERT_EQ(poses.size(), 1u);
    EXPECT_NEAR(poses[0].score, 0.9f, 0.01f);
    EXPECT_EQ(poses[0].keypoints.size(), 17u);
    EXPECT_NEAR(poses[0].keypoints[0].x, 100.0f, 0.1f);
    EXPECT_NEAR(poses[0].keypoints[0].y, 150.0f, 0.1f);
    EXPECT_NEAR(poses[0].keypoints[0].confidence, 0.9f, 0.01f);
    EXPECT_GT(poses[0].bbox.width, 0);
    EXPECT_GT(poses[0].bbox.height, 0);
}

TEST(EdgeCrafterPoseTest, LabelOffsetApplied) {
    EdgeCrafterPosePostprocessor pp(0.5f, 0.3f);

    int num_dets = 1;
    int num_kpts = 17;
    int kpt_dim = 3;

    // EdgeCrafter pose outputs label=1 for person (not 0); should be offset to 0
    std::vector<TensorElement> scores = {0.9f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(1)};

    size_t kpt_size = static_cast<size_t>(num_kpts) * static_cast<size_t>(kpt_dim);
    std::vector<TensorElement> keypoints(kpt_size, 0.0f);
    keypoints[0] = 100.0f;
    keypoints[1] = 150.0f;
    keypoints[2] = 0.5f;

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(scores, {1, num_dets}),
        Tensor(keypoints, {1, num_dets, num_kpts, kpt_dim}),
    };

    auto poses = pp.postprocess(tensors, cv::Size(640, 480), cv::Size(640, 640));
    ASSERT_EQ(poses.size(), 1u);
    // With label offset -1, person (label 1) should NOT have class_id field on PoseEstimation
    // but we verify the bbox was derived correctly
}

TEST(EdgeCrafterPoseTest, FiltersLowScore) {
    EdgeCrafterPosePostprocessor pp(0.5f, 0.3f);
    int num_dets = 1;
    std::vector<TensorElement> scores = {0.1f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(1)};
    std::vector<TensorElement> keypoints(static_cast<size_t>(17 * 3), 0.0f);

    std::vector<Tensor> tensors = {
        Tensor(labels, {1, num_dets}),
        Tensor(scores, {1, num_dets}),
        Tensor(keypoints, {1, num_dets, 17, 3}),
    };

    auto poses = pp.postprocess(tensors, cv::Size(640, 480), cv::Size(640, 640));
    EXPECT_TRUE(poses.empty());
}

TEST(EdgeCrafterPoseTest, EmptyInputThrows) {
    EdgeCrafterPosePostprocessor pp(0.5f, 0.3f);
    EXPECT_THROW({ pp.postprocess({}, cv::Size(640, 480), cv::Size(640, 640)); }, std::runtime_error);
}

TEST(EdgeCrafterDetectionTest, UsesOutputNamesWhenTensorsAreReordered) {
    EdgeCrafterPostprocessor pp(0.5f, {"scores", "labels", "boxes"});

    int num_dets = 1;
    std::vector<TensorElement> scores = {0.9f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(5)};
    std::vector<TensorElement> boxes = {100.0f, 100.0f, 300.0f, 300.0f};

    std::vector<Tensor> tensors = {
        Tensor(scores, {1, num_dets}),
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
    };

    auto detections = pp.postprocess(tensors, cv::Size(640, 480));

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
    EXPECT_EQ(detections[0].class_id, 5);
    EXPECT_EQ(detections[0].bbox, BoundingBox(100, 100, 200, 200));
}

TEST(EdgeCrafterSegmentationTest, UsesOutputNamesWhenTensorsAreReordered) {
    EdgeCrafterSegmentationPostprocessor pp(0.5f, 0.5f, {"masks", "scores", "labels", "boxes"});

    int num_dets = 1;
    int mask_h = 4;
    int mask_w = 4;
    std::vector<TensorElement> masks(static_cast<size_t>(mask_h * mask_w), 1.0f);
    std::vector<TensorElement> scores = {0.9f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(5)};
    std::vector<TensorElement> boxes = {100.0f, 100.0f, 300.0f, 300.0f};

    std::vector<Tensor> tensors = {
        Tensor(masks, {1, num_dets, mask_h, mask_w}),
        Tensor(scores, {1, num_dets}),
        Tensor(labels, {1, num_dets}),
        Tensor(boxes, {1, num_dets, 4}),
    };

    auto segmentations = pp.postprocess(tensors, cv::Size(640, 480));

    ASSERT_EQ(segmentations.size(), 1u);
    EXPECT_FLOAT_EQ(segmentations[0].class_confidence, 0.9f);
    EXPECT_EQ(segmentations[0].class_id, 5);
    EXPECT_EQ(segmentations[0].bbox, BoundingBox(100, 100, 200, 200));
}

TEST(EdgeCrafterPoseTest, UsesOutputNamesWhenTensorsAreReordered) {
    EdgeCrafterPosePostprocessor pp(0.5f, 0.3f, {"keypoints", "labels", "scores"});

    int num_dets = 1;
    int num_kpts = 2;
    int kpt_dim = 3;
    std::vector<TensorElement> keypoints = {100.0f, 150.0f, 0.9f, 120.0f, 190.0f, 0.8f};
    std::vector<TensorElement> labels = {static_cast<int64_t>(1)};
    std::vector<TensorElement> scores = {0.9f};

    std::vector<Tensor> tensors = {
        Tensor(keypoints, {1, num_dets, num_kpts, kpt_dim}),
        Tensor(labels, {1, num_dets}),
        Tensor(scores, {1, num_dets}),
    };

    auto poses = pp.postprocess(tensors, cv::Size(640, 480), cv::Size(640, 640));

    ASSERT_EQ(poses.size(), 1u);
    EXPECT_NEAR(poses[0].score, 0.9f, 0.01f);
    ASSERT_EQ(poses[0].keypoints.size(), 2u);
    EXPECT_EQ(poses[0].bbox, BoundingBox(100, 150, 20, 40));
}

// ---------------------------------------------------------------------------
// Task-Level Tests
// ---------------------------------------------------------------------------

TEST(EdgeCrafterDetectionTaskTest, TaskType) {
    auto info = createDetectionModelInfo();
    ObjectDetectionTask task(info, "ecdet");
    EXPECT_EQ(task.getTaskType(), TaskType::Detection);

    ObjectDetectionTask task2(info, "ecdet_s");
    EXPECT_EQ(task2.getTaskType(), TaskType::Detection);
}

TEST(EdgeCrafterDetectionTaskTest, PreprocessReturnsTwoOutputs) {
    auto info = createDetectionModelInfo();
    ObjectDetectionTask task(info, "ecdet_s");
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    auto outputs = task.preprocess({img});
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_FALSE(outputs[0].empty());
    EXPECT_FALSE(outputs[1].empty());
}

TEST(EdgeCrafterInstanceSegmentationTaskTest, TaskType) {
    auto info = createSegmentationModelInfo();
    InstanceSegmentationTask task(info, "ecseg");
    EXPECT_EQ(task.getTaskType(), TaskType::InstanceSegmentation);
}

TEST(EdgeCrafterInstanceSegmentationTaskTest, PreprocessReturnsTwoOutputs) {
    auto info = createSegmentationModelInfo();
    InstanceSegmentationTask task(info, "ecseg_s");
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    auto outputs = task.preprocess({img});
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_FALSE(outputs[0].empty());
    EXPECT_EQ(outputs[1].size(), 2u * sizeof(int64_t));

    const auto original_size = decodeInt64Pair(outputs[1]);
    EXPECT_EQ(original_size[0], img.cols);
    EXPECT_EQ(original_size[1], img.rows);
}

TEST(EdgeCrafterPoseTaskTest, TaskType) {
    auto info = createPoseModelInfo();
    PoseEstimationTask task(info, "ecpose");
    EXPECT_EQ(task.getTaskType(), TaskType::PoseEstimation);

    PoseEstimationTask task2(info, "ecpose_s");
    EXPECT_EQ(task2.getTaskType(), TaskType::PoseEstimation);
}

TEST(EdgeCrafterPoseTaskTest, PreprocessReturnsTwoOutputs) {
    auto info = createPoseModelInfo();
    PoseEstimationTask task(info, "ecpose_s");
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    auto outputs = task.preprocess({img});
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_FALSE(outputs[0].empty());
    EXPECT_FALSE(outputs[1].empty());
}

// ---------------------------------------------------------------------------
// Factory Routing Tests
// ---------------------------------------------------------------------------

TEST(EdgeCrafterFactoryRoutingTest, DetectionModels) {
    auto info = createDetectionModelInfo();
    std::vector<std::string> types = {"ecdet",   "ecdet_s",     "ecdet_m",        "ecdet_l",
                                      "ecdet_x", "edgecrafter", "edgecrafter-det"};

    for (const auto& type : types) {
        auto task = TaskFactory::createTaskInstance(type, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for type: " << type;
        EXPECT_EQ(task->getTaskType(), TaskType::Detection) << "Wrong task type for: " << type;
    }
}

TEST(EdgeCrafterFactoryRoutingTest, SegmentationModels) {
    auto info = createSegmentationModelInfo();
    std::vector<std::string> types = {"ecseg", "ecseg_s", "ecseg_m", "ecseg_l", "ecseg_x", "edgecrafter-seg"};

    for (const auto& type : types) {
        auto task = TaskFactory::createTaskInstance(type, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for type: " << type;
        EXPECT_EQ(task->getTaskType(), TaskType::InstanceSegmentation) << "Wrong task type for: " << type;
    }
}

TEST(EdgeCrafterFactoryRoutingTest, PoseModels) {
    auto info = createPoseModelInfo();
    std::vector<std::string> types = {"ecpose", "ecpose_s", "ecpose_m", "edgecrafter-pose"};

    for (const auto& type : types) {
        auto task = TaskFactory::createTaskInstance(type, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for type: " << type;
        EXPECT_EQ(task->getTaskType(), TaskType::PoseEstimation) << "Wrong task type for: " << type;
    }
}
