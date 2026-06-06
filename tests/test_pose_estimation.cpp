#include "neuriplo/tasks/pose_estimation/pose_estimation_task.hpp"
#include "neuriplo/tasks/pose_estimation/vit_pose_postprocessor.hpp"
#include "neuriplo/tasks/pose_estimation/yolo_pose_postprocessor.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace neuriplo_tasks;

// ---------------------------------------------------------------------------
// ViTPose tests
// ---------------------------------------------------------------------------

TEST(ViTPosePostprocessorTest, BasicPostprocess) {
    ViTPosePostprocessor processor;

    int num_joints = 17;
    int heatmap_h = 64;
    int heatmap_w = 48;

    std::vector<TensorElement> heatmap_data(num_joints * heatmap_h * heatmap_w, 0.0f);

    // Set a peak for the first joint (Nose) at (24, 32) in heatmap
    int x_peak = 24;
    int y_peak = 32;
    heatmap_data[0 * heatmap_h * heatmap_w + y_peak * heatmap_w + x_peak] = 1.0f;

    std::vector<Tensor> tensors = {Tensor(heatmap_data, {1, num_joints, heatmap_h, heatmap_w})};

    cv::Size original_size(1920, 1080);
    cv::Size input_size(192, 256);

    auto results = processor.postprocess(tensors, original_size, input_size);

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].keypoints.size(), static_cast<size_t>(num_joints));

    // kp.x = 24 * 1920 / 48 = 960
    // kp.y = 32 * 1080 / 64 = 540
    EXPECT_NEAR(results[0].keypoints[0].x, 960.0f, 0.1f);
    EXPECT_NEAR(results[0].keypoints[0].y, 540.0f, 0.1f);
    EXPECT_NEAR(results[0].keypoints[0].confidence, 1.0f, 0.01f);
}

TEST(PoseEstimationTaskTest, TaskTypeViTPose) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 256, 192}};
    info.input_formats = {"FORMAT_NCHW"};

    PoseEstimationTask task(info, "vitpose");
    EXPECT_EQ(task.getTaskType(), TaskType::PoseEstimation);
}

// ---------------------------------------------------------------------------
// YOLO pose postprocessor tests
// ---------------------------------------------------------------------------

// Helper: build a minimal [1, 56, N] tensor with one high-confidence detection
static std::vector<Tensor> makeYolov8PoseTensor(int anchors, int num_kpts, float cx, float cy, float w, float h,
                                                float conf) {
    int channels = 4 + 1 + num_kpts * 3; // 56 for 17 kpts
    std::vector<TensorElement> data(static_cast<size_t>(channels * anchors), 0.0f);

    // First anchor: set bbox + confidence
    // Layout [channels, anchors] → data[ch * anchors + 0]
    data[static_cast<size_t>(0 * anchors + 0)] = cx;
    data[static_cast<size_t>(1 * anchors + 0)] = cy;
    data[static_cast<size_t>(2 * anchors + 0)] = w;
    data[static_cast<size_t>(3 * anchors + 0)] = h;
    data[static_cast<size_t>(4 * anchors + 0)] = conf;

    // First keypoint at (cx, cy) with confidence 0.9
    data[static_cast<size_t>(5 * anchors + 0)] = cx;
    data[static_cast<size_t>(6 * anchors + 0)] = cy;
    data[static_cast<size_t>(7 * anchors + 0)] = 0.9f;

    return {Tensor(data, {1, channels, anchors})};
}

TEST(YoloPosePostprocessorTest, DetectsPersonYolov8Format) {
    // Input 640x640, frame 1280x720
    cv::Size input_size(640, 640);
    cv::Size frame_size(1280, 720);

    YoloPosePostprocessor pp(input_size, 0.25f, 0.45f);

    // Place detection at model center (320, 320), 200x200 box
    auto tensors = makeYolov8PoseTensor(8400, 17, 320.0f, 320.0f, 200.0f, 200.0f, 0.9f);
    auto results = pp.postprocess(tensors, frame_size, input_size);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_NEAR(results[0].score, 0.9f, 0.01f);
    EXPECT_EQ(static_cast<int>(results[0].keypoints.size()), 17);
}

TEST(YoloPosePostprocessorTest, FiltersLowConfidence) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640);

    YoloPosePostprocessor pp(input_size, 0.25f, 0.45f);

    // conf = 0.1 < threshold 0.25
    auto tensors = makeYolov8PoseTensor(8400, 17, 320.0f, 320.0f, 100.0f, 100.0f, 0.1f);
    auto results = pp.postprocess(tensors, frame_size, input_size);

    EXPECT_TRUE(results.empty());
}

TEST(YoloPosePostprocessorTest, EmptyTensorReturnsEmpty) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640);

    YoloPosePostprocessor pp(input_size, 0.25f, 0.45f);
    auto results = pp.postprocess({}, frame_size, input_size);
    EXPECT_TRUE(results.empty());
}

TEST(YoloPosePostprocessorTest, BboxPopulated) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640); // square, no letterbox padding

    YoloPosePostprocessor pp(input_size, 0.25f, 0.45f);

    // cx=320, cy=320, w=100, h=100 → x_min=270, y_min=270, w=100, h=100 (same scale, no pad)
    auto tensors = makeYolov8PoseTensor(8400, 17, 320.0f, 320.0f, 100.0f, 100.0f, 0.8f);
    auto results = pp.postprocess(tensors, frame_size, input_size);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_GT(results[0].bbox.width, 0);
    EXPECT_GT(results[0].bbox.height, 0);
}

// ---------------------------------------------------------------------------
// TaskFactory routing test for YOLO pose
// ---------------------------------------------------------------------------

TEST(PoseEstimationTaskTest, TaskTypeYoloPose) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};

    // Various YOLO pose name variants
    for (const auto& name : {"yolov8pose", "yolov5pose", "yolov11pose", "yolo26pose"}) {
        PoseEstimationTask task(info, name);
        EXPECT_EQ(task.getTaskType(), TaskType::PoseEstimation) << "Failed for: " << name;
    }
}
