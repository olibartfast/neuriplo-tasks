#include "neuriplo/tasks/core/batch_preprocess.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

namespace {

ModelInfo classificationModelInfo(int max_batch_size) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.input_types = {neuriplo_tasks::PixelType::Float32};
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
    info.input_types = {neuriplo_tasks::PixelType::Float32};
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
    request.images = {neuriplo_tasks::vision_test::makeImage(120, 100, 3, 0)};

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
    request.images = {neuriplo_tasks::vision_test::makeImage(90, 80, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(64, 64, 3, 1)};

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
    request.images = {neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(150, 200, 3, 0)};

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
    request.images = {neuriplo_tasks::vision_test::makeImage(10, 10, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(10, 10, 3, 0)};

    EXPECT_THROW(batchPreprocess(*task, request), std::invalid_argument);
}
