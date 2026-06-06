#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/gaussian_splatting/gaussian_splatting_task.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

class GaussianSplattingTest : public ::testing::Test {
  protected:
    ModelInfo createModelInfo(int h = 256, int w = 256) {
        ModelInfo info;
        info.input_shapes = {{1, 3, h, w}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"image"};
        info.output_names = {"gaussians"};
        info.input_types = {CV_32F};
        return info;
    }

    // Build a flat tensor of shape [G, 14] filled with increasing floats
    static std::vector<TensorElement> makeTensorData(int num_gaussians) {
        std::vector<TensorElement> data;
        data.reserve(static_cast<size_t>(num_gaussians) * 14u);
        for (int i = 0; i < num_gaussians * 14; ++i) {
            data.emplace_back(static_cast<float>(i) * 0.01f);
        }
        return data;
    }
};

// ---------------------------------------------------------------------------
// Construction / factory
// ---------------------------------------------------------------------------

TEST_F(GaussianSplattingTest, TaskFactoryCreatesTask) {
    auto info = createModelInfo();
    auto task = TaskFactory::createTaskInstance("lgm", info);

    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::GaussianSplatting);
}

TEST_F(GaussianSplattingTest, TaskFactoryCreatesTaskAliasSplat) {
    auto info = createModelInfo();
    auto task = TaskFactory::createTaskInstance("gaussiansplatting", info);

    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::GaussianSplatting);
}

TEST_F(GaussianSplattingTest, TaskFactoryCreatesGrmTask) {
    auto info = createModelInfo();
    auto task = TaskFactory::createTaskInstance("grm", info);

    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::GaussianSplatting);
}

TEST_F(GaussianSplattingTest, RequiredFramesIsCorrect) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");
    EXPECT_EQ(task.getRequiredFrames(), 4);
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

TEST_F(GaussianSplattingTest, PreprocessReturnsTensorData) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");

    cv::Mat img = cv::Mat::zeros(300, 400, CV_8UC3);
    auto outputs = task.preprocess({img});

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_FALSE(outputs[0].empty());

    // Expected buffer size: 1 * 3 * 256 * 256 * sizeof(float)
    const size_t expected = 1u * 3u * 256u * 256u * sizeof(float);
    EXPECT_EQ(outputs[0].size(), expected);
}

// ---------------------------------------------------------------------------
// Postprocessing — [G, 14] layout
// ---------------------------------------------------------------------------

TEST_F(GaussianSplattingTest, PostprocessFlat2DLayout) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");

    const int G = 8;
    Tensor t(makeTensorData(G), {G, 14});
    auto results = task.postprocess(cv::Size(256, 256), {t});

    ASSERT_EQ(results.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<GaussianSplatting>(results[0]));

    const auto& splat = std::get<GaussianSplatting>(results[0]);
    EXPECT_EQ(splat.num_gaussians, G);
    ASSERT_EQ(static_cast<int>(splat.gaussians.size()), G);
}

// ---------------------------------------------------------------------------
// Postprocessing — [N, G, 14] batched layout
// ---------------------------------------------------------------------------

TEST_F(GaussianSplattingTest, PostprocessBatched3DLayout) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");

    const int N = 1;
    const int G = 16;
    Tensor t(makeTensorData(N * G), {N, G, 14});
    auto results = task.postprocess(cv::Size(256, 256), {t});

    ASSERT_EQ(results.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<GaussianSplatting>(results[0]));

    const auto& splat = std::get<GaussianSplatting>(results[0]);
    EXPECT_EQ(splat.num_gaussians, G);
}

// ---------------------------------------------------------------------------
// Postprocessing — field values
// ---------------------------------------------------------------------------

TEST_F(GaussianSplattingTest, PostprocessFieldValuesCorrect) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");

    // Build a 1-Gaussian tensor with known values
    std::vector<TensorElement> data(14);
    for (int i = 0; i < 14; ++i) {
        data[static_cast<size_t>(i)] = static_cast<float>(i);
    }
    Tensor t(data, {1, 14});
    auto results = task.postprocess(cv::Size(256, 256), {t});

    ASSERT_EQ(results.size(), 1u);
    const auto& splat = std::get<GaussianSplatting>(results[0]);
    ASSERT_EQ(splat.num_gaussians, 1);

    const Gaussian3D& g = splat.gaussians[0];
    EXPECT_FLOAT_EQ(g.x, 0.0f);
    EXPECT_FLOAT_EQ(g.y, 1.0f);
    EXPECT_FLOAT_EQ(g.z, 2.0f);
    EXPECT_FLOAT_EQ(g.scale_x, 3.0f);
    EXPECT_FLOAT_EQ(g.scale_y, 4.0f);
    EXPECT_FLOAT_EQ(g.scale_z, 5.0f);
    EXPECT_FLOAT_EQ(g.rot_w, 6.0f);
    EXPECT_FLOAT_EQ(g.rot_x, 7.0f);
    EXPECT_FLOAT_EQ(g.rot_y, 8.0f);
    EXPECT_FLOAT_EQ(g.rot_z, 9.0f);
    EXPECT_FLOAT_EQ(g.opacity, 10.0f);
    EXPECT_FLOAT_EQ(g.sh_r, 11.0f);
    EXPECT_FLOAT_EQ(g.sh_g, 12.0f);
    EXPECT_FLOAT_EQ(g.sh_b, 13.0f);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_F(GaussianSplattingTest, PostprocessEmptyTensorReturnsEmpty) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");

    auto results = task.postprocess(cv::Size(256, 256), {});
    EXPECT_TRUE(results.empty());
}

TEST_F(GaussianSplattingTest, PostprocessTruncatedDataReturnsEmpty) {
    auto info = createModelInfo();
    GaussianSplattingTask task(info, "lgm");

    // Tensor claims [10, 14] but data only has 5 elements
    Tensor t({TensorElement{0.f}, TensorElement{1.f}, TensorElement{2.f}, TensorElement{3.f}, TensorElement{4.f}},
             {10, 14});
    auto results = task.postprocess(cv::Size(256, 256), {t});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(std::get<GaussianSplatting>(results[0]).num_gaussians, 0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
