#include "neuriplo/tasks/pose_estimation/pose_estimation_task.hpp"
#include "neuriplo/tasks/pose_estimation/rfdetr_pose_postprocessor.hpp"
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

// ---------------------------------------------------------------------------
// RF-DETR pose postprocessor tests
// ---------------------------------------------------------------------------

// Helper: build synthetic 3-tensor RF-DETR keypoint output
// dets: [1, 1, 4], labels: [1, 1, num_classes], keypoints: [1, 1, num_classes*num_kpts, 8]
static std::vector<Tensor> makeRfDetrPoseTensors(float det_cx, float det_cy, float det_w, float det_h,
                                                 float person_logit, int num_classes, int num_kpts,
                                                 const std::vector<float>& keypoint_data) {
    std::vector<TensorElement> dets(4);
    dets[0] = det_cx;
    dets[1] = det_cy;
    dets[2] = det_w;
    dets[3] = det_h;

    std::vector<TensorElement> labels(static_cast<size_t>(num_classes), -10.0f);
    if (num_classes > 1) {
        labels[1] = person_logit;
    }

    int keypoint_dim2 = num_classes * num_kpts;
    size_t kp_total = static_cast<size_t>(keypoint_dim2 * 8);
    std::vector<TensorElement> kps(kp_total, 0.0f);
    for (size_t i = 0; i < keypoint_data.size(); ++i) {
        kps[i] = keypoint_data[i];
    }

    return {Tensor(dets, {1, 1, 4}), Tensor(labels, {1, 1, static_cast<int64_t>(num_classes)}),
            Tensor(kps, {1, 1, static_cast<int64_t>(keypoint_dim2), 8})};
}

TEST(RfDetrPosePostprocessorTest, DetectsSinglePerson) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640);

    RfDetrPosePostprocessor pp(input_size, 0.25f, 0.5f, {0, 17});

    int num_classes = 2;
    int num_kpts = 17;
    int keypoint_dim2 = num_classes * num_kpts;

    std::vector<float> kp_data(static_cast<size_t>(keypoint_dim2 * 8), 0.0f);

    // Fill person keypoints at offset num_kpts*8 (class 1)
    for (int k = 0; k < num_kpts; ++k) {
        size_t base = static_cast<size_t>(num_kpts + k) * 8;
        kp_data[base + 0] = 0.5f;
        kp_data[base + 1] = 0.5f;
        kp_data[base + 2] = 5.0f;
        kp_data[base + 3] = 5.0f;
        kp_data[base + 4] = 0.0f;
        kp_data[base + 5] = 0.0f;
        kp_data[base + 6] = 0.0f;
        kp_data[base + 7] = 0.0f;
    }

    auto tensors = makeRfDetrPoseTensors(0.5f, 0.5f, 0.25f, 0.25f, 10.0f, num_classes, num_kpts, kp_data);
    auto results = pp.postprocess(tensors, frame_size, input_size);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_GT(results[0].score, 0.0f);
    EXPECT_GT(results[0].bbox.width, 0);
    EXPECT_GT(results[0].bbox.height, 0);
    EXPECT_EQ(static_cast<int>(results[0].keypoints.size()), num_kpts);

    // Each keypoint at frame center (320, 320)
    for (const auto& kp : results[0].keypoints) {
        EXPECT_NEAR(kp.x, 320.0f, 0.1f);
        EXPECT_NEAR(kp.y, 320.0f, 0.1f);
        EXPECT_GT(kp.confidence, 0.5f);
        EXPECT_GT(kp.visibility, 0.5f);
    }
}

TEST(RfDetrPosePostprocessorTest, CovarianceRoundTrips) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640);

    int num_kpts = 1;
    RfDetrPosePostprocessor pp(input_size, 0.25f, 0.5f, {0, num_kpts});

    int num_classes = 2;
    int keypoint_dim2 = num_classes * num_kpts;

    std::vector<float> kp_data(static_cast<size_t>(keypoint_dim2 * 8), 0.0f);

    // Single person keypoint with non-trivial Cholesky: L11=2, L21=1, L22=2
    // → det=16, cov00=5/16=0.3125, cov01=-2/16=-0.125, cov10=-0.125, cov11=4/16=0.25
    size_t base = static_cast<size_t>(num_kpts) * 8;
    kp_data[base + 0] = 0.3f;
    kp_data[base + 1] = 0.6f;
    kp_data[base + 2] = 5.0f;
    kp_data[base + 3] = 5.0f;
    kp_data[base + 4] = std::log(2.0f);
    kp_data[base + 5] = 1.0f;
    kp_data[base + 6] = std::log(2.0f);
    kp_data[base + 7] = 0.0f;

    auto tensors = makeRfDetrPoseTensors(0.5f, 0.5f, 0.25f, 0.25f, 10.0f, num_classes, num_kpts, kp_data);
    auto results = pp.postprocess(tensors, frame_size, input_size);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].keypoints.size(), 1u);

    const auto& kp = results[0].keypoints[0];
    EXPECT_NEAR(kp.x, 0.3f * 640.0f, 0.1f);
    EXPECT_NEAR(kp.y, 0.6f * 640.0f, 0.1f);

    EXPECT_NEAR(kp.covariance[0], 0.3125f, 0.01f);
    EXPECT_NEAR(kp.covariance[1], -0.125f, 0.01f);
    EXPECT_NEAR(kp.covariance[2], -0.125f, 0.01f);
    EXPECT_NEAR(kp.covariance[3], 0.25f, 0.01f);

    // Covariance diagonal entries should be positive
    EXPECT_GT(kp.covariance[0], 0.0f);
    EXPECT_GT(kp.covariance[3], 0.0f);
}

TEST(RfDetrPosePostprocessorTest, FiltersBelowThreshold) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640);

    RfDetrPosePostprocessor pp(input_size, 0.9f, 0.5f, {0, 17});

    int num_classes = 2;
    int num_kpts = 17;
    int keypoint_dim2 = num_classes * num_kpts;
    std::vector<float> kp_data(static_cast<size_t>(keypoint_dim2 * 8), 0.0f);

    auto tensors = makeRfDetrPoseTensors(0.5f, 0.5f, 0.25f, 0.25f, -5.0f, num_classes, num_kpts, kp_data);
    auto results = pp.postprocess(tensors, frame_size, input_size);

    EXPECT_TRUE(results.empty());
}

TEST(RfDetrPosePostprocessorTest, EmptyTensorReturnsEmpty) {
    cv::Size input_size(640, 640);
    cv::Size frame_size(640, 640);

    RfDetrPosePostprocessor pp(input_size, 0.25f, 0.5f, {0, 17});
    EXPECT_THROW(pp.postprocess({}, frame_size, input_size), std::runtime_error);
}

// ---------------------------------------------------------------------------
// TaskFactory routing test for RF-DETR pose
// ---------------------------------------------------------------------------

TEST(PoseEstimationTaskTest, TaskTypeRfDetrPose) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};

    for (const auto& name : {"rfdetrpose", "rfdetrkeypoint", "rfdetr+kpt", "rfdetr_pose"}) {
        PoseEstimationTask task(info, name);
        EXPECT_EQ(task.getTaskType(), TaskType::PoseEstimation) << "Failed for: " << name;
    }
}
