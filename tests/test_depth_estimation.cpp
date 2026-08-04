#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/depth_estimation/depth_estimation_task.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

class DepthEstimationTest : public ::testing::Test {
  protected:
    ModelInfo createModelInfo() {
        ModelInfo info;
        info.input_shapes = {{1, 3, 518, 518}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"pixel_values"};
        info.output_names = {"predicted_depth"};
        info.input_types = {neuriplo_tasks::PixelType::Float32};
        return info;
    }
};

TEST_F(DepthEstimationTest, PreprocessReturnsTensorData) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    neuriplo_tasks::Image img = neuriplo_tasks::vision_test::makeImage(400, 300, 3, 0);
    auto outputs = task.preprocess({img});

    ASSERT_EQ(outputs.size(), 1);
    EXPECT_FALSE(outputs[0].empty());
}

TEST_F(DepthEstimationTest, PreprocessRejectsEmptyImage) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    EXPECT_THROW(task.preprocess({neuriplo_tasks::Image()}), std::invalid_argument);
}

TEST_F(DepthEstimationTest, YoloDepthPreprocessUsesYoloNormalizationAndPadding) {
    auto model_info = createModelInfo();
    model_info.input_shapes = {{1, 3, 2, 2}};
    DepthEstimationTask task(model_info, "yolo26n-depth");

    neuriplo_tasks::Image image = neuriplo_tasks::vision_test::makeImage(2, 1, 3, 255);
    const auto outputs = task.preprocess({image});

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].size(), 3U * 2U * 2U * sizeof(float));
    const auto* values = reinterpret_cast<const float*>(outputs[0].data());
    for (size_t channel = 0; channel < 3U; ++channel) {
        const size_t offset = channel * 4U;
        EXPECT_FLOAT_EQ(values[offset], 1.0f);
        EXPECT_FLOAT_EQ(values[offset + 1U], 1.0f);
        EXPECT_FLOAT_EQ(values[offset + 2U], 114.0f / 255.0f);
        EXPECT_FLOAT_EQ(values[offset + 3U], 114.0f / 255.0f);
    }
}

TEST_F(DepthEstimationTest, PostprocessConvertsDepthMap) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    const int batch = 1;
    const int height = 4;
    const int width = 5;

    std::vector<TensorElement> data;
    data.reserve(static_cast<size_t>(batch * height * width));

    for (int i = 0; i < batch * height * width; ++i) {
        data.emplace_back(static_cast<float>(i));
    }

    Tensor output_tensor(data, {batch, height, width});

    auto results = task.postprocess(neuriplo_tasks::Size(10, 8), {output_tensor});

    ASSERT_EQ(results.size(), 1);
    ASSERT_TRUE(std::holds_alternative<DepthEstimation>(results[0]));

    const auto& depth = std::get<DepthEstimation>(results[0]);
    EXPECT_EQ(depth.depth.rows(), 8);
    EXPECT_EQ(depth.depth.cols(), 10);
    EXPECT_EQ(depth.depth.pixelType(), neuriplo_tasks::PixelType::Float32);
    EXPECT_EQ(depth.normalized_depth.pixelType(), neuriplo_tasks::PixelType::Float32);
    EXPECT_LE(depth.min_depth, depth.max_depth);
}

TEST_F(DepthEstimationTest, PostprocessEmptyTensorReturnsEmptyResult) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    auto results = task.postprocess(neuriplo_tasks::Size(10, 8), {});
    EXPECT_TRUE(results.empty());
}

TEST_F(DepthEstimationTest, YoloDepthPostprocessRemovesLetterboxPadding) {
    auto model_info = createModelInfo();
    model_info.input_shapes = {{1, 3, 6, 6}};
    DepthEstimationTask task(model_info, "yolo26n-depth");

    std::vector<TensorElement> data(36, 100.0f);
    for (int column = 0; column < 6; ++column) {
        data[12U + static_cast<size_t>(column)] = 1.0f;
        data[18U + static_cast<size_t>(column)] = 2.0f;
    }

    const auto results = task.postprocess(neuriplo_tasks::Size(6, 2), {Tensor(data, {1, 1, 6, 6})});

    ASSERT_EQ(results.size(), 1);
    const auto& depth = std::get<DepthEstimation>(results[0]);
    EXPECT_EQ(depth.depth.rows(), 2);
    EXPECT_EQ(depth.depth.cols(), 6);
    EXPECT_FLOAT_EQ(depth.min_depth, 1.0f);
    EXPECT_FLOAT_EQ(depth.max_depth, 2.0f);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
