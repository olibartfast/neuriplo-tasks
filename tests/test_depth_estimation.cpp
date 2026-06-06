#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/depth_estimation/depth_estimation_task.hpp"

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
        info.input_types = {CV_32F};
        return info;
    }
};

TEST_F(DepthEstimationTest, PreprocessReturnsTensorData) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    cv::Mat img = cv::Mat::zeros(300, 400, CV_8UC3);
    auto outputs = task.preprocess({img});

    ASSERT_EQ(outputs.size(), 1);
    EXPECT_FALSE(outputs[0].empty());
}

TEST_F(DepthEstimationTest, PreprocessRejectsEmptyImage) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    EXPECT_THROW(task.preprocess({cv::Mat()}), std::invalid_argument);
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

    auto results = task.postprocess(cv::Size(10, 8), {output_tensor});

    ASSERT_EQ(results.size(), 1);
    ASSERT_TRUE(std::holds_alternative<DepthEstimation>(results[0]));

    const auto& depth = std::get<DepthEstimation>(results[0]);
    EXPECT_EQ(depth.depth.rows(), 8);
    EXPECT_EQ(depth.depth.cols(), 10);
    EXPECT_EQ(depth.depth.type(), CV_32FC1);
    EXPECT_EQ(depth.normalized_depth.type(), CV_32FC1);
    EXPECT_LE(depth.min_depth, depth.max_depth);
}

TEST_F(DepthEstimationTest, PostprocessEmptyTensorReturnsEmptyResult) {
    auto model_info = createModelInfo();
    DepthEstimationTask task(model_info, "depth_anything_v2");

    auto results = task.postprocess(cv::Size(10, 8), {});
    EXPECT_TRUE(results.empty());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
