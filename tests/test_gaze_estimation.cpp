#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/gaze_estimation/gaze_estimation_postprocessor.hpp"
#include "neuriplo/tasks/gaze_estimation/gaze_estimation_task.hpp"
#include "vision_test_utils.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace neuriplo_tasks;

namespace {

ModelInfo makeGazeInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"pixel_values"};
    info.output_names = {"gaze"};
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

} // namespace

TEST(GazePostprocessorTest, AnglesDecodeToDirection) {
    GazePostprocessor pp;
    const float pitch = 0.1f;
    const float yaw = 0.2f;
    auto results = pp.postprocess({makeTensor({pitch, yaw}, {1, 2})});
    ASSERT_EQ(results.size(), 1U);
    EXPECT_NEAR(results[0].pitch, pitch, 1e-5f);
    EXPECT_NEAR(results[0].yaw, yaw, 1e-5f);
    const auto expected = GazePostprocessor::directionFromPitchYaw(pitch, yaw);
    EXPECT_NEAR(results[0].direction[0], expected[0], 1e-5f);
    EXPECT_NEAR(results[0].direction[1], expected[1], 1e-5f);
    EXPECT_NEAR(results[0].direction[2], expected[2], 1e-5f);
    const float norm = std::sqrt(results[0].direction[0] * results[0].direction[0] +
                                 results[0].direction[1] * results[0].direction[1] +
                                 results[0].direction[2] * results[0].direction[2]);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(GazePostprocessorTest, ZeroAnglesLookForward) {
    GazePostprocessor pp;
    auto results = pp.postprocess({makeTensor({0.0f, 0.0f}, {1, 2})});
    ASSERT_EQ(results.size(), 1U);
    EXPECT_NEAR(results[0].direction[0], 0.0f, 1e-6f);
    EXPECT_NEAR(results[0].direction[1], 0.0f, 1e-6f);
    EXPECT_NEAR(results[0].direction[2], -1.0f, 1e-6f);
}

TEST(GazePostprocessorTest, MatchesPtgazeAngleConvention) {
    // ETH-XGaze / OpenGaze normalized-camera convention (cf. ptgaze
    // gaze_angles_to_vector): horizontal flips negate yaw only.
    const auto pos = GazePostprocessor::directionFromPitchYaw(0.1f, 0.5f);
    EXPECT_NEAR(pos[0], -std::cos(0.1f) * std::sin(0.5f), 1e-6f);
    EXPECT_NEAR(pos[1], -std::sin(0.1f), 1e-6f);
    EXPECT_NEAR(pos[2], -std::cos(0.1f) * std::cos(0.5f), 1e-6f);
    const auto neg = GazePostprocessor::directionFromPitchYaw(0.1f, -0.5f);
    EXPECT_NEAR(neg[0], -pos[0], 1e-6f);
    EXPECT_NEAR(neg[1], pos[1], 1e-6f);
    EXPECT_NEAR(neg[2], pos[2], 1e-6f);
}

TEST(GazePostprocessorTest, VectorNormalizesAndDerivesAngles) {
    GazePostprocessor pp;
    auto results = pp.postprocess({makeTensor({0.0f, 0.0f, -2.0f}, {1, 3})});
    ASSERT_EQ(results.size(), 1U);
    EXPECT_NEAR(results[0].direction[2], -1.0f, 1e-6f);
    EXPECT_NEAR(results[0].pitch, 0.0f, 1e-5f);
    EXPECT_NEAR(results[0].yaw, 0.0f, 1e-5f);
}

TEST(GazePostprocessorTest, VectorRoundTripsAngles) {
    const float pitch = -0.15f;
    const float yaw = 0.35f;
    const auto dir = GazePostprocessor::directionFromPitchYaw(pitch, yaw);
    GazePostprocessor pp;
    auto results = pp.postprocess({makeTensor({dir[0], dir[1], dir[2]}, {1, 3})});
    ASSERT_EQ(results.size(), 1U);
    EXPECT_NEAR(results[0].pitch, pitch, 1e-5f);
    EXPECT_NEAR(results[0].yaw, yaw, 1e-5f);
}

TEST(GazePostprocessorTest, BatchAndConfidenceTensor) {
    GazePostprocessor pp;
    std::vector<Tensor> tensors = {makeTensor({0.0f, 0.0f, 0.1f, 0.2f}, {2, 2}), makeTensor({0.9f, 0.4f}, {2})};
    auto results = pp.postprocess(tensors);
    ASSERT_EQ(results.size(), 2U);
    EXPECT_NEAR(results[0].confidence, 0.9f, 1e-6f);
    EXPECT_NEAR(results[1].confidence, 0.4f, 1e-6f);
    EXPECT_NEAR(results[1].pitch, 0.1f, 1e-6f);
}

TEST(GazePostprocessorTest, MalformedShapesReturnEmpty) {
    GazePostprocessor pp;
    EXPECT_TRUE(pp.postprocess({}).empty());
    EXPECT_TRUE(pp.postprocess({makeTensor({0.1f, 0.2f, 0.3f, 0.4f}, {1, 4})}).empty());
    EXPECT_TRUE(pp.postprocess({makeTensor({0.1f}, {1, 1, 2})}).empty());
}

TEST(GazeEstimationTaskTest, PreprocessIsImageNetNchw) {
    GazeEstimationTask task(makeGazeInfo(), "ethxgaze");
    EXPECT_EQ(task.getTaskType(), TaskType::GazeEstimation);
    auto outputs = task.preprocess({vision_test::makeImage(64, 48, 3, 128)});
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs[0].size(), 3U * 224U * 224U * sizeof(float));
}

TEST(GazeEstimationTaskTest, PostprocessReturnsGazeResult) {
    GazeEstimationTask task(makeGazeInfo(), "gazetr");
    auto results = task.postprocess(Size(64, 48), {makeTensor({0.0f, 0.0f}, {1, 2})});
    ASSERT_EQ(results.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<GazeEstimation>(results[0]));
}

TEST(GazeEstimationTaskTest, FactoryRoutesAliases) {
    auto info = makeGazeInfo();
    for (const char* alias : {"gaze", "ethxgaze", "gazetr", "ETH-XGaze", "GazeTR"}) {
        auto task = TaskFactory::createTaskInstance(alias, info);
        ASSERT_NE(task, nullptr) << alias;
        EXPECT_EQ(task->getTaskType(), TaskType::GazeEstimation) << alias;
    }
}
