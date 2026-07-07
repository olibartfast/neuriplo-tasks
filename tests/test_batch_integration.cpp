#include "neuriplo/tasks/core/batch_postprocess.hpp"
#include "neuriplo/tasks/core/batch_preprocess.hpp"
#include "neuriplo/tasks/core/task_config.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

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

ModelInfo detectionModelInfo() {
    ModelInfo info;
    info.input_shapes = {{2, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"images"};
    info.output_names = {"output0"};
    info.input_types = {neuriplo_tasks::PixelType::Float32};
    info.batch_size_ = 2;
    info.max_batch_size_ = 2;
    return info;
}

Tensor makeClassificationTensor() {
    const std::vector<float> logits = {0.1f, 0.2f, 0.3f, 5.0f, 0.0f, 1.0f, 4.0f, 0.5f};
    std::vector<TensorElement> data;
    data.reserve(logits.size());
    for (float v : logits) {
        data.emplace_back(v);
    }
    return Tensor(std::move(data), {2, 4});
}

Tensor makeYoloTensor(int batch, int anchors, int num_classes) {
    const int channels = 4 + num_classes;
    std::vector<TensorElement> output(static_cast<size_t>(batch * channels * anchors), 0.0f);

    auto set_detection = [&](int batch_index, int class_id, float score) {
        const size_t base =
            static_cast<size_t>(batch_index) * static_cast<size_t>(channels) * static_cast<size_t>(anchors);
        output[base + static_cast<size_t>(0 * anchors + 0)] = 320.0f;
        output[base + static_cast<size_t>(1 * anchors + 0)] = 320.0f;
        output[base + static_cast<size_t>(2 * anchors + 0)] = 100.0f;
        output[base + static_cast<size_t>(3 * anchors + 0)] = 100.0f;
        output[base + static_cast<size_t>((4 + class_id) * anchors + 0)] = score;
    };

    set_detection(0, 0, 0.95f);
    set_detection(1, 2, 0.95f);

    return Tensor(std::move(output), {batch, channels, anchors});
}

} // namespace

TEST(BatchIntegrationTest, ClassificationPreprocessPostprocessPipeline) {
    TaskConfig config;
    config.apply_softmax = false;

    auto task = TaskFactory::createTaskInstance("resnet50", classificationModelInfo(), config);
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {neuriplo_tasks::vision_test::makeImage(110, 90, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(64, 64, 3, 1)};

    const auto pre = batchPreprocess(*task, request);
    ASSERT_EQ(pre.batch_size, 2);
    ASSERT_EQ(pre.buffers.size(), 2u);
    EXPECT_TRUE(imageBatchSizeMatches(request, pre.batch_size));

    const neuriplo_tasks::Size frame_size(90, 110);
    const auto post = batchPostprocess(*task, frame_size, {makeClassificationTensor()}, pre.batch_size);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_TRUE(postprocessResultsMatchBatchSize(post));
    ASSERT_EQ(post.results.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<Classification>(post.results[0]));
    EXPECT_EQ(std::get<Classification>(post.results[0]).class_id, 3);
    EXPECT_EQ(std::get<Classification>(post.results[1]).class_id, 2);
}

TEST(BatchIntegrationTest, ObjectDetectionPreprocessPostprocessPipeline) {
    auto task = TaskFactory::createTaskInstance("yolov8", detectionModelInfo());
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(150, 200, 3, 0)};

    const auto pre = batchPreprocess(*task, request);
    ASSERT_EQ(pre.batch_size, 2);
    ASSERT_EQ(pre.buffers.size(), 2u);

    const neuriplo_tasks::Size frame_size(640, 640);
    const Tensor tensor = makeYoloTensor(2, 8, 4);
    const auto post = batchPostprocess(*task, frame_size, {tensor}, pre.batch_size);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_FALSE(post.results.empty());

    bool saw_detection = false;
    for (const auto& result : post.results) {
        if (std::holds_alternative<Detection>(result)) {
            saw_detection = true;
            break;
        }
    }
    EXPECT_TRUE(saw_detection);
}
