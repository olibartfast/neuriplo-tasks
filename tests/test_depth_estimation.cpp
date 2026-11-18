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

TEST_F(DepthEstimationTest, DepthAnything3PreprocessUsesImageNetNormalization) {
    auto model_info = createModelInfo();
    model_info.input_shapes = {{1, 3, 504, 504}};
    DepthEstimationTask task(model_info, "da3-small");

    neuriplo_tasks::Image image = neuriplo_tasks::vision_test::makeImage(504, 504, 3, 255);
    const auto outputs = task.preprocess({image});

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].size(), 3U * 504U * 504U * sizeof(float));
    const auto* values = reinterpret_cast<const float*>(outputs[0].data());
    // White pixel through [0,1] + ImageNet normalization, BGR input swapped to
    // RGB: R channel uses means[0]/stds[0], B channel uses means[2]/stds[2].
    EXPECT_FLOAT_EQ(values[0], (1.0f - 0.485f) / 0.229f);
    EXPECT_FLOAT_EQ(values[2U * 504U * 504U], (1.0f - 0.406f) / 0.225f);
}

TEST_F(DepthEstimationTest, DepthAnything3FallsBackTo504OnEmptyShapes) {
    ModelInfo model_info;
    model_info.input_formats = {"FORMAT_NCHW"};
    DepthEstimationTask task(model_info, "depth-anything-3");

    neuriplo_tasks::Image image = neuriplo_tasks::vision_test::makeImage(640, 480, 3, 128);
    const auto outputs = task.preprocess({image});

    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(), 3U * 504U * 504U * sizeof(float));
}

TEST_F(DepthEstimationTest, DepthAnything3PostprocessConvertsSingleBatchMap) {
    auto model_info = createModelInfo();
    model_info.input_shapes = {{1, 3, 504, 504}};
    DepthEstimationTask task(model_info, "da3mono-large");

    std::vector<TensorElement> data;
    for (int i = 0; i < 20; ++i) {
        data.emplace_back(static_cast<float>(i));
    }

    const auto results = task.postprocess(neuriplo_tasks::Size(5, 4), {Tensor(data, {1, 4, 5})});

    ASSERT_EQ(results.size(), 1);
    ASSERT_TRUE(std::holds_alternative<DepthEstimation>(results[0]));
    const auto& depth = std::get<DepthEstimation>(results[0]);
    EXPECT_EQ(depth.depth.rows(), 4);
    EXPECT_EQ(depth.depth.cols(), 5);
    EXPECT_FLOAT_EQ(depth.min_depth, 0.0f);
    EXPECT_FLOAT_EQ(depth.max_depth, 19.0f);
}

TEST_F(DepthEstimationTest, DepthAnything3PostprocessResizesToFrame) {
    auto model_info = createModelInfo();
    model_info.input_shapes = {{1, 3, 504, 504}};
    DepthEstimationTask task(model_info, "da3-small");

    std::vector<TensorElement> data;
    for (int i = 0; i < 20; ++i) {
        data.emplace_back(static_cast<float>(i));
    }

    const auto results = task.postprocess(neuriplo_tasks::Size(10, 8), {Tensor(data, {1, 4, 5})});

    ASSERT_EQ(results.size(), 1);
    const auto& depth = std::get<DepthEstimation>(results[0]);
    EXPECT_EQ(depth.depth.rows(), 8);
    EXPECT_EQ(depth.depth.cols(), 10);
    EXPECT_LE(depth.min_depth, depth.max_depth);
}

TEST_F(DepthEstimationTest, DepthAnything3MetricAndNestedNamesAreRejected) {
    auto model_info = createModelInfo();
    EXPECT_THROW(DepthEstimationTask(model_info, "da3metric-large"), std::invalid_argument);
    EXPECT_THROW(DepthEstimationTask(model_info, "da3-nested-giant-large"), std::invalid_argument);
    EXPECT_THROW(DepthEstimationTask(model_info, "depth-anything-3-metric"), std::invalid_argument);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
