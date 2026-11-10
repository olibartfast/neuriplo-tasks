#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_postprocessor.hpp"
#include "neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_task.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

namespace {

constexpr int kAnchors = 84;
constexpr int kCorners = 8;

ModelInfo makeMonoInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"images"};
    info.output_names = {"output0"};
    info.input_types = {PixelType::Float32};
    return info;
}

Tensor makeTensor(std::vector<float> values, std::vector<int64_t> shape) {
    std::vector<TensorElement> data;
    data.reserve(values.size());
    for (float v : values) {
        data.emplace_back(v);
    }
    return Tensor(data, shape);
}

// Channels-first [1, C, N] YOLOv8/v11 pose layout: cx, cy, w, h (input
// pixels), nc class scores, then 8 x/y/conf corner triples (input pixels).
// Corners are emitted ground-first (0-3 y=400, 4-7 y=100), matching the
// released UrbanOmniDetect checkpoints.
std::vector<Tensor> makeChannelsFirst(float conf = 0.9f, int num_classes = 1, int good_anchor = 0) {
    const int channels = 4 + num_classes + kCorners * 3;
    std::vector<float> data(static_cast<size_t>(channels * kAnchors), 0.0f);
    auto at = [&](int ch, int anchor) -> float& { return data[static_cast<size_t>(ch * kAnchors + anchor)]; };
    at(0, good_anchor) = 320.0f;
    at(1, good_anchor) = 320.0f;
    at(2, good_anchor) = 200.0f;
    at(3, good_anchor) = 200.0f;
    for (int c = 0; c < num_classes; ++c) {
        at(4 + c, good_anchor) = (c == 0) ? conf : conf - 0.5f;
    }
    for (int k = 0; k < kCorners; ++k) {
        const bool ground = k < 4;
        at(4 + num_classes + k * 3 + 0, good_anchor) = 220.0f + static_cast<float>(k) * 20.0f;
        at(4 + num_classes + k * 3 + 1, good_anchor) = ground ? 400.0f : 100.0f;
        at(4 + num_classes + k * 3 + 2, good_anchor) = 0.8f;
    }
    return {makeTensor(data, {1, channels, kAnchors})};
}

} // namespace

TEST(Monocular3DPostprocessorTest, DecodesChannelsFirstInGroundFirstOrder) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f);
    auto results = pp.postprocess(makeChannelsFirst(), Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    const auto& det = results[0];
    EXPECT_EQ(det.bbox.x, 220);
    EXPECT_EQ(det.bbox.y, 220);
    EXPECT_EQ(det.bbox.width, 200);
    EXPECT_EQ(det.bbox.height, 200);
    EXPECT_EQ(det.class_id, 0);
    EXPECT_FLOAT_EQ(det.class_confidence, 0.9f);
    EXPECT_FLOAT_EQ(det.geometry_confidence, 0.8f);
    ASSERT_EQ(det.projected_corners.size(), 8U);
    // Released checkpoints emit ground-first; order is preserved.
    EXPECT_FLOAT_EQ(det.projected_corners[0].y, 400.0f);
    EXPECT_FLOAT_EQ(det.projected_corners[0].x, 220.0f);
    EXPECT_FLOAT_EQ(det.projected_corners[4].y, 100.0f);
    EXPECT_FLOAT_EQ(det.projected_corners[0].confidence, 0.8f);
}

TEST(Monocular3DPostprocessorTest, TopFirstRemapIsOptIn) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f, false, true);
    const int channels = 4 + 1 + kCorners * 3;
    std::vector<float> data(static_cast<size_t>(channels * kAnchors), 0.0f);
    auto at = [&](int ch, int anchor) -> float& { return data[static_cast<size_t>(ch * kAnchors + anchor)]; };
    at(0, 0) = 320.0f;
    at(1, 0) = 320.0f;
    at(2, 0) = 100.0f;
    at(3, 0) = 100.0f;
    at(4, 0) = 0.9f;
    for (int k = 0; k < kCorners; ++k) {
        at(5 + k * 3 + 0, 0) = 300.0f;
        at(5 + k * 3 + 1, 0) = (k < 4) ? 100.0f : 400.0f;
        at(5 + k * 3 + 2, 0) = 0.7f;
    }
    auto results = pp.postprocess({makeTensor(data, {1, channels, kAnchors})}, Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_FLOAT_EQ(results[0].projected_corners[0].y, 400.0f);
    EXPECT_FLOAT_EQ(results[0].projected_corners[4].y, 100.0f);
}

TEST(Monocular3DPostprocessorTest, SupportsRowsLayoutWithObjectness) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f);
    constexpr int rows = 84;
    constexpr int channels = 4 + 1 + 1 + kCorners * 3;
    std::vector<float> data(static_cast<size_t>(rows * channels), 0.0f);
    auto at = [&](int row, int ch) -> float& { return data[static_cast<size_t>(row * channels + ch)]; };
    at(0, 0) = 320.0f;
    at(0, 1) = 320.0f;
    at(0, 2) = 200.0f;
    at(0, 3) = 200.0f;
    at(0, 4) = 0.9f;
    at(0, 5) = 0.8f;
    for (int k = 0; k < kCorners; ++k) {
        at(0, 6 + k * 3 + 0) = 300.0f;
        at(0, 6 + k * 3 + 1) = 300.0f;
        at(0, 6 + k * 3 + 2) = 0.5f;
    }
    auto results = pp.postprocess({makeTensor(data, {1, rows, channels})}, Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_NEAR(results[0].class_confidence, 0.72f, 1e-5f);
    EXPECT_EQ(results[0].bbox.width, 200);
}

TEST(Monocular3DPostprocessorTest, SupportsEndToEndRows) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f, true);
    constexpr int rows = 300;
    constexpr int channels = 6 + kCorners * 3;
    std::vector<float> data(static_cast<size_t>(rows * channels), 0.0f);
    auto at = [&](int row, int ch) -> float& { return data[static_cast<size_t>(row * channels + ch)]; };
    at(0, 0) = 100.0f;
    at(0, 1) = 120.0f;
    at(0, 2) = 300.0f;
    at(0, 3) = 520.0f;
    at(0, 4) = 0.9f;
    at(0, 5) = 2.0f;
    for (int k = 0; k < kCorners; ++k) {
        at(0, 6 + k * 3 + 0) = 150.0f;
        at(0, 6 + k * 3 + 1) = 160.0f;
        at(0, 6 + k * 3 + 2) = 0.8f;
    }
    auto results = pp.postprocess({makeTensor(data, {1, rows, channels})}, Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].bbox.x, 100);
    EXPECT_EQ(results[0].bbox.y, 120);
    EXPECT_EQ(results[0].bbox.width, 200);
    EXPECT_EQ(results[0].bbox.height, 400);
    EXPECT_EQ(results[0].class_id, 2);
    EXPECT_FLOAT_EQ(results[0].class_confidence, 0.9f);
}

TEST(Monocular3DPostprocessorTest, SupportsTwoValueCornersWithoutConfidence) {
    // Real v1 export layout [B, 4+3+16, N] (kpt_shape [8, 2]): no per-corner
    // confidence, so corner confidence falls back to the detection score.
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f);
    const int channels = 4 + 3 + kCorners * 2;
    std::vector<float> data(static_cast<size_t>(channels * kAnchors), 0.0f);
    auto at = [&](int ch, int anchor) -> float& { return data[static_cast<size_t>(ch * kAnchors + anchor)]; };
    at(0, 0) = 320.0f;
    at(1, 0) = 320.0f;
    at(2, 0) = 200.0f;
    at(3, 0) = 200.0f;
    at(4, 0) = 0.9f;
    at(5, 0) = 0.1f;
    at(6, 0) = 0.1f;
    for (int k = 0; k < kCorners; ++k) {
        at(7 + k * 2 + 0, 0) = 300.0f;
        at(7 + k * 2 + 1, 0) = (k < 4) ? 400.0f : 100.0f;
    }
    auto results = pp.postprocess({makeTensor(data, {1, channels, kAnchors})}, Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].class_id, 0);
    EXPECT_FLOAT_EQ(results[0].projected_corners[0].y, 400.0f);
    EXPECT_FLOAT_EQ(results[0].projected_corners[0].confidence, 0.9f);
    EXPECT_FLOAT_EQ(results[0].geometry_confidence, 0.9f);
}

TEST(Monocular3DPostprocessorTest, MultiClassTakesArgmax) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f);
    auto tensors = makeChannelsFirst(0.9f, 3);
    // Second class wins.
    const int channels = 4 + 3 + kCorners * 3;
    std::vector<float> raw;
    raw.reserve(static_cast<size_t>(channels * kAnchors));
    for (const auto& e : tensors[0].data) {
        raw.push_back(std::get<float>(e));
    }
    raw[static_cast<size_t>(4 * kAnchors + 0)] = 0.1f;
    raw[static_cast<size_t>(5 * kAnchors + 0)] = 0.95f;
    auto results = pp.postprocess({makeTensor(raw, {1, channels, kAnchors})}, Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].class_id, 1);
    EXPECT_FLOAT_EQ(results[0].class_confidence, 0.95f);
}

TEST(Monocular3DPostprocessorTest, LetterboxPaddingIsRemoved) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f);
    auto results = pp.postprocess(makeChannelsFirst(), Size(640, 320));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_NEAR(results[0].bbox.x + results[0].bbox.width / 2, 320.0, 2.0);
}

TEST(Monocular3DPostprocessorTest, NmsDeduplicatesOverlappingBoxes) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.25f, 0.45f);
    const int channels = 4 + 1 + kCorners * 3;
    std::vector<float> data(static_cast<size_t>(channels * kAnchors), 0.0f);
    auto at = [&](int ch, int anchor) -> float& { return data[static_cast<size_t>(ch * kAnchors + anchor)]; };
    for (int a = 0; a < 2; ++a) {
        at(0, a) = 320.0f;
        at(1, a) = 320.0f;
        at(2, a) = 200.0f;
        at(3, a) = 200.0f;
        at(4, a) = 0.9f - static_cast<float>(a) * 0.1f;
        for (int k = 0; k < kCorners; ++k) {
            at(5 + k * 3 + 0, a) = 300.0f;
            at(5 + k * 3 + 1, a) = 300.0f;
            at(5 + k * 3 + 2, a) = 0.8f;
        }
    }
    auto results = pp.postprocess({makeTensor(data, {1, channels, kAnchors})}, Size(640, 640));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_FLOAT_EQ(results[0].class_confidence, 0.9f);
}

TEST(Monocular3DPostprocessorTest, ThresholdEmptyAndMalformed) {
    Monocular3DPostprocessor pp(Size(640, 640), 0.5f, 0.45f);
    EXPECT_TRUE(pp.postprocess(makeChannelsFirst(0.1f), Size(640, 640)).empty());
    EXPECT_TRUE(pp.postprocess({}, Size(640, 640)).empty());
    // Wrong corner count is not a cuboid head.
    const int bad_channels = 4 + 1 + 7 * 3;
    EXPECT_TRUE(
        pp.postprocess({makeTensor(std::vector<float>(bad_channels * kAnchors, 0.0f), {1, bad_channels, kAnchors})},
                       Size(640, 640))
            .empty());
}

TEST(Monocular3DTaskTest, PreprocessPostprocessAndFactory) {
    Monocular3DDetectionTask task(makeMonoInfo(), "urbanomnidetect");
    EXPECT_EQ(task.getTaskType(), TaskType::Monocular3DDetection);
    auto outputs = task.preprocess({vision_test::makeImage(640, 480, 3, 10)});
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_FALSE(outputs[0].empty());
    auto results = task.postprocess(Size(640, 480), makeChannelsFirst());
    ASSERT_EQ(results.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<Monocular3DDetection>(results[0]));

    Monocular3DDetectionTask v2task(makeMonoInfo(), "urbanomnidetect2");
    EXPECT_EQ(v2task.getTaskType(), TaskType::Monocular3DDetection);

    for (const char* alias : {"urbanomni", "urbanomnidetect", "urbanomnidetect2", "monocular3d", "mono3d"}) {
        auto routed = TaskFactory::createTaskInstance(alias, makeMonoInfo());
        ASSERT_NE(routed, nullptr) << alias;
        EXPECT_EQ(routed->getTaskType(), TaskType::Monocular3DDetection) << alias;
    }
}
