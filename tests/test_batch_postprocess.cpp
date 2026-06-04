#include "vision-core/core/batch_postprocess.hpp"
#include "vision-core/core/task_config.hpp"
#include "vision-core/core/task_factory.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vision_core;

namespace {

ModelInfo depthModelInfo(int max_batch_size) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 518, 518}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"pixel_values"};
    info.output_names = {"predicted_depth"};
    info.input_types = {CV_32F};
    info.max_batch_size_ = max_batch_size;
    info.batch_size_ = 1;
    return info;
}

Tensor makeDepthTensor(int batch, int height, int width) {
    std::vector<TensorElement> data;
    data.reserve(static_cast<size_t>(batch * height * width));
    for (int i = 0; i < batch * height * width; ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    return Tensor(std::move(data), {batch, height, width});
}

bool resultsEqual(const std::vector<Result>& lhs, const std::vector<Result>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].index() != rhs[i].index()) {
            return false;
        }
        if (!std::holds_alternative<DepthEstimation>(lhs[i]) || !std::holds_alternative<DepthEstimation>(rhs[i])) {
            return false;
        }
        const auto& left_depth = std::get<DepthEstimation>(lhs[i]);
        const auto& right_depth = std::get<DepthEstimation>(rhs[i]);
        if (left_depth.depth.size() != right_depth.depth.size() ||
            left_depth.depth.type() != right_depth.depth.type()) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST(BatchPostprocessTest, SingleBatchMatchesDirectPostprocess) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo(2));
    ASSERT_NE(task, nullptr);

    const cv::Size frame_size(40, 30);
    const std::vector<Tensor> tensors = {makeDepthTensor(1, 4, 5)};

    const auto direct = task->postprocess(frame_size, tensors);
    const auto wrapped = batchPostprocess(*task, frame_size, tensors, 1);

    EXPECT_TRUE(resultsEqual(wrapped.results, direct));
    EXPECT_EQ(wrapped.batch_size, 1);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(wrapped));
}

TEST(BatchPostprocessTest, TwoBatchIndicesRoundTrip) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo(2));
    ASSERT_NE(task, nullptr);

    const cv::Size frame_size(64, 48);
    const std::vector<Tensor> tensors = {makeDepthTensor(2, 3, 4)};

    const auto direct = task->postprocess(frame_size, tensors);
    const auto wrapped = batchPostprocess(*task, frame_size, tensors, 2);

    EXPECT_EQ(direct.size(), 2u);
    EXPECT_TRUE(resultsEqual(wrapped.results, direct));
    EXPECT_EQ(wrapped.batch_size, 2);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(wrapped));
}

TEST(BatchPostprocessTest, GaussianSplattingKeepsSingleAggregateResult) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 256, 256}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"image"};
    info.output_names = {"gaussians"};
    info.input_types = {CV_32F};
    info.max_batch_size_ = 4;

    auto task = TaskFactory::createTaskInstance("lgm", info);
    ASSERT_NE(task, nullptr);

    const int num_gaussians = 8;
    std::vector<TensorElement> data;
    data.reserve(static_cast<size_t>(num_gaussians) * 14u);
    for (int i = 0; i < num_gaussians * 14; ++i) {
        data.emplace_back(static_cast<float>(i));
    }
    const std::vector<Tensor> tensors = {Tensor(std::move(data), {num_gaussians, 14})};

    const auto wrapped = batchPostprocess(*task, cv::Size(256, 256), tensors, 4);

    EXPECT_EQ(wrapped.results.size(), 1u);
    EXPECT_EQ(wrapped.batch_size, 4);
    EXPECT_FALSE(postprocessResultsMatchBatchSize(wrapped));
    ASSERT_TRUE(std::holds_alternative<GaussianSplatting>(wrapped.results[0]));
}

TEST(BatchPostprocessTest, RejectsNonPositiveBatchSize) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo(1));
    ASSERT_NE(task, nullptr);

    const std::vector<Tensor> tensors = {makeDepthTensor(1, 2, 2)};
    EXPECT_THROW(batchPostprocess(*task, cv::Size(10, 8), tensors, 0), std::invalid_argument);
}

TEST(BatchPostprocessTest, RejectsBatchExceedingMaxBatchSize) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo(1));
    ASSERT_NE(task, nullptr);

    const std::vector<Tensor> tensors = {makeDepthTensor(2, 2, 2)};
    EXPECT_THROW(batchPostprocess(*task, cv::Size(10, 8), tensors, 2), std::invalid_argument);
}

TEST(BatchPostprocessTest, ClassificationBatchSizeTwo) {
    ModelInfo info;
    info.input_shapes = {{2, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.input_types = {CV_32F};
    info.max_batch_size_ = 2;

    TaskConfig config;
    config.apply_softmax = false;

    auto task = TaskFactory::createTaskInstance("resnet50", info, config);
    ASSERT_NE(task, nullptr);

    std::vector<TensorElement> data = {0.1f, 0.2f, 3.0f, 0.0f, 2.0f, 4.0f, 0.5f, 0.1f};
    const std::vector<Tensor> tensors = {Tensor(std::move(data), {2, 4})};

    const auto wrapped = batchPostprocess(*task, cv::Size(100, 80), tensors, 2);

    EXPECT_EQ(wrapped.batch_size, 2);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(wrapped));
}

TEST(BatchPostprocessTest, EmptyTensorsReturnsEmptyResults) {
    auto task = TaskFactory::createTaskInstance("depthanythingv2", depthModelInfo(2));
    ASSERT_NE(task, nullptr);

    const auto wrapped = batchPostprocess(*task, cv::Size(10, 8), {}, 2);

    EXPECT_TRUE(wrapped.results.empty());
    EXPECT_EQ(wrapped.batch_size, 2);
}
