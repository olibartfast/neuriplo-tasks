#include "vision-core/core/batch_postprocess.hpp"
#include "vision-core/core/batch_preprocess.hpp"
#include "vision-core/core/task_factory.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vision_core;

namespace {

ModelInfo depthModelInfo() {
    ModelInfo info;
    info.input_shapes = {{2, 3, 518, 518}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"pixel_values"};
    info.output_names = {"predicted_depth"};
    info.input_types = {CV_32F};
    info.batch_size_ = 2;
    info.max_batch_size_ = 2;
    return info;
}

Tensor makeDepthTensor(int batch, int height, int width) {
    std::vector<TensorElement> data;
    data.reserve(static_cast<size_t>(batch * height * width));
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < height * width; ++i) {
            data.emplace_back(static_cast<float>(b * 100 + i));
        }
    }
    return Tensor(std::move(data), {batch, height, width});
}

} // namespace

TEST(DepthEstimationBatchTest, PostprocessReturnsTwoDepthMaps) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo());
    ASSERT_NE(task, nullptr);

    const std::vector<Tensor> tensors = {makeDepthTensor(2, 3, 4)};
    const auto results = task->postprocess(cv::Size(40, 30), tensors);

    ASSERT_EQ(results.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<DepthEstimation>(results[0]));
    ASSERT_TRUE(std::holds_alternative<DepthEstimation>(results[1]));
}

TEST(DepthEstimationBatchTest, BatchHelpersRoundTrip) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo());
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {cv::Mat::zeros(120, 160, CV_8UC3), cv::Mat::ones(100, 100, CV_8UC3)};

    const auto pre = batchPreprocess(*task, request);
    const auto post = batchPostprocess(*task, cv::Size(160, 120), {makeDepthTensor(2, 3, 4)}, pre.batch_size);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(post));
}
