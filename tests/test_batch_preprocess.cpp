#include "vision-core/core/batch_preprocess.hpp"
#include "vision-core/core/task_factory.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vision_core;

namespace {

ModelInfo classificationModelInfo(int max_batch_size) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.input_types = {CV_32F};
    info.max_batch_size_ = max_batch_size;
    info.batch_size_ = 1;
    return info;
}

ModelInfo detectionModelInfo(int max_batch_size) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"images"};
    info.output_names = {"output0"};
    info.input_types = {CV_32F};
    info.max_batch_size_ = max_batch_size;
    info.batch_size_ = 1;
    return info;
}

bool buffersEqual(const std::vector<std::vector<uint8_t>>& lhs, const std::vector<std::vector<uint8_t>>& rhs) {
    return lhs == rhs;
}

} // namespace

TEST(BatchPreprocessTest, SingleImageMatchesDirectPreprocess) {
    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(1));
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {cv::Mat::zeros(100, 120, CV_8UC3)};

    const auto direct = task->preprocess(request.images);
    const auto wrapped = batchPreprocess(*task, request);

    EXPECT_TRUE(buffersEqual(wrapped.buffers, direct));
    EXPECT_EQ(wrapped.batch_size, 1);
    EXPECT_TRUE(imageBatchSizeMatches(request, wrapped.batch_size));
}

TEST(BatchPreprocessTest, TwoImagesReturnsTwoBuffers) {
    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(2));
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {cv::Mat::zeros(80, 90, CV_8UC3), cv::Mat::ones(64, 64, CV_8UC3)};

    const auto wrapped = batchPreprocess(*task, request);

    EXPECT_EQ(wrapped.batch_size, 2);
    EXPECT_EQ(wrapped.buffers.size(), 2u);
    EXPECT_FALSE(wrapped.buffers[0].empty());
    EXPECT_FALSE(wrapped.buffers[1].empty());
    EXPECT_TRUE(imageBatchSizeMatches(request, wrapped.batch_size));
}

TEST(BatchPreprocessTest, ObjectDetectionTwoImages) {
    auto task = TaskFactory::createTaskInstance("yolov8", detectionModelInfo(2));
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {cv::Mat::zeros(100, 100, CV_8UC3), cv::Mat::zeros(200, 150, CV_8UC3)};

    const auto direct = task->preprocess(request.images);
    const auto wrapped = batchPreprocess(*task, request);

    EXPECT_TRUE(buffersEqual(wrapped.buffers, direct));
    EXPECT_EQ(wrapped.batch_size, 2);
    EXPECT_EQ(wrapped.buffers.size(), 2u);
}

TEST(BatchPreprocessTest, RejectsEmptyImages) {
    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(1));
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    EXPECT_THROW(batchPreprocess(*task, request), std::invalid_argument);
}

TEST(BatchPreprocessTest, RejectsBatchExceedingMaxBatchSize) {
    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(1));
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {cv::Mat::zeros(10, 10, CV_8UC3), cv::Mat::zeros(10, 10, CV_8UC3)};

    EXPECT_THROW(batchPreprocess(*task, request), std::invalid_argument);
}
