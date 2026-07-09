#include "neuriplo/tasks/classification/classification_postprocessor.hpp"
#include "neuriplo/tasks/core/batch_postprocess.hpp"
#include "neuriplo/tasks/core/batch_preprocess.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace neuriplo_tasks;

namespace {

ModelInfo classificationModelInfo() {
    ModelInfo info;
    info.input_shapes = {{2, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.input_types = {neuriplo_tasks::PixelType::Float32};
    info.batch_size_ = 2;
    info.max_batch_size_ = 2;
    return info;
}

Tensor makeBatchedLogitsTensor() {
    // batch 0: class 3 wins; batch 1: class 1 wins
    const std::vector<float> logits = {0.1f, 0.2f, 0.3f, 5.0f, 0.0f, 1.0f, 4.0f, 0.5f};
    std::vector<TensorElement> data;
    data.reserve(logits.size());
    for (float v : logits) {
        data.emplace_back(v);
    }
    return Tensor(std::move(data), {2, 4});
}

} // namespace

TEST(ClassificationBatchTest, PostprocessorReturnsOneResultPerBatchIndex) {
    DefaultClassificationPostprocessor processor(5, false);

    const auto results = processor.postprocess(makeBatchedLogitsTensor().data, makeBatchedLogitsTensor().shape);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].class_id, 3);
    EXPECT_EQ(results[1].class_id, 2);
}

TEST(ClassificationBatchTest, TaskPostprocessBatchSizeTwo) {
    TaskConfig config;
    config.top_k = 5;
    config.apply_softmax = false;

    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(), config);
    ASSERT_NE(task, nullptr);

    const std::vector<Tensor> tensors = {makeBatchedLogitsTensor()};
    const auto results = task->postprocess(neuriplo_tasks::Size(100, 80), tensors);

    ASSERT_EQ(results.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<Classification>(results[0]));
    EXPECT_EQ(std::get<Classification>(results[0]).class_id, 3);
    EXPECT_EQ(std::get<Classification>(results[1]).class_id, 2);
}

TEST(ClassificationBatchTest, PreprocessRejectsEmptyImage) {
    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo());
    ASSERT_NE(task, nullptr);

    EXPECT_THROW(task->preprocess({neuriplo_tasks::Image()}), std::invalid_argument);
}

TEST(ClassificationBatchTest, BatchPreprocessPostprocessRoundTrip) {
    TaskConfig config;
    config.apply_softmax = false;

    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(), config);
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {neuriplo_tasks::vision_test::makeImage(110, 90, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(64, 64, 3, 1)};

    const auto pre = batchPreprocess(*task, request);
    EXPECT_EQ(pre.batch_size, 2);

    const auto post =
        batchPostprocess(*task, neuriplo_tasks::Size(90, 110), {makeBatchedLogitsTensor()}, pre.batch_size);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(post));
    ASSERT_EQ(post.results.size(), 2u);
}

TEST(ClassificationBatchTest, SingleBatchUnchangedTopK) {
    DefaultClassificationPostprocessor processor(3, false);

    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 3.0f, 0.1f};
    std::vector<TensorElement> data;
    for (float v : logits) {
        data.emplace_back(v);
    }

    const auto results = processor.postprocess(data, {1, 5});

    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].class_id, 3);
}
