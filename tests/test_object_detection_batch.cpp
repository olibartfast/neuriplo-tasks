#include "vision-core/core/batch_postprocess.hpp"
#include "vision-core/core/batch_preprocess.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/yolo_postprocessor.hpp"

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vision_core;

namespace {

ModelInfo detectionModelInfo() {
    ModelInfo info;
    info.input_shapes = {{2, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"images"};
    info.output_names = {"output0"};
    info.input_types = {CV_32F};
    info.batch_size_ = 2;
    info.max_batch_size_ = 2;
    return info;
}

ModelInfo multiInputDetectionModelInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 320, 544}, {1, 2}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"images", "orig_target_sizes"};
    info.output_names = {"scores", "boxes"};
    info.input_types = {CV_32F, CV_32S};
    info.batch_size_ = 1;
    info.max_batch_size_ = 1;
    return info;
}

std::array<int64_t, 2> decodeInt64Pair(const std::vector<uint8_t>& buffer) {
    std::array<int64_t, 2> values{};
    std::memcpy(values.data(), buffer.data(), values.size() * sizeof(int64_t));
    return values;
}

Tensor makeBatchedYoloTensor(int batch, int anchors, int num_classes) {
    const int channels = 4 + num_classes;
    std::vector<TensorElement> output(static_cast<size_t>(batch * channels * anchors), 0.0f);

    auto set_detection = [&](int batch_index, int class_id, float cx, float cy, float w, float h, float score) {
        const size_t base =
            static_cast<size_t>(batch_index) * static_cast<size_t>(channels) * static_cast<size_t>(anchors);
        output[base + static_cast<size_t>(0 * anchors + 0)] = cx;
        output[base + static_cast<size_t>(1 * anchors + 0)] = cy;
        output[base + static_cast<size_t>(2 * anchors + 0)] = w;
        output[base + static_cast<size_t>(3 * anchors + 0)] = h;
        output[base + static_cast<size_t>((4 + class_id) * anchors + 0)] = score;
    };

    set_detection(0, 0, 320.0f, 320.0f, 100.0f, 100.0f, 0.95f);
    set_detection(1, 2, 320.0f, 320.0f, 100.0f, 100.0f, 0.95f);

    return Tensor(std::move(output), {batch, channels, anchors});
}

} // namespace

TEST(ObjectDetectionBatchTest, YoloStandardDecodesEachBatchSlice) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, cv::Size(640, 640), 0.25f, 0.45f);

    const int anchors = 8;
    const int num_classes = 4;
    const Tensor tensor = makeBatchedYoloTensor(2, anchors, num_classes);

    const auto detections = processor.postprocess({tensor}, cv::Size(640, 640));

    ASSERT_GE(detections.size(), 2u);

    bool saw_class0 = false;
    bool saw_class2 = false;
    for (const auto& det : detections) {
        if (det.class_id == 0) {
            saw_class0 = true;
        }
        if (det.class_id == 2) {
            saw_class2 = true;
        }
    }
    EXPECT_TRUE(saw_class0);
    EXPECT_TRUE(saw_class2);
}

TEST(ObjectDetectionBatchTest, TaskFactoryBatchPreprocessPostprocess) {
    auto task = TaskFactory::createTaskInstance("yolov8", detectionModelInfo());
    ASSERT_NE(task, nullptr);

    BatchRequest request;
    request.images = {cv::Mat::zeros(100, 100, CV_8UC3), cv::Mat::zeros(200, 150, CV_8UC3)};

    const auto pre = batchPreprocess(*task, request);
    EXPECT_EQ(pre.batch_size, 2);
    EXPECT_EQ(pre.buffers.size(), 2u);

    const Tensor tensor = makeBatchedYoloTensor(2, 8, 4);
    const auto direct = task->postprocess(cv::Size(640, 640), {tensor});
    const auto post = batchPostprocess(*task, cv::Size(640, 640), {tensor}, pre.batch_size);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_FALSE(direct.empty());
    EXPECT_EQ(post.results.size(), direct.size());
}

TEST(ObjectDetectionBatchTest, RtDetrPreprocessUsesModelInputSizeMetadata) {
    auto task = TaskFactory::createTaskInstance("rtdetr", multiInputDetectionModelInfo());
    ASSERT_NE(task, nullptr);

    const auto buffers = task->preprocess({cv::Mat::zeros(123, 456, CV_8UC3)});

    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_FALSE(buffers[0].empty());
    ASSERT_EQ(buffers[1].size(), 2u * sizeof(int64_t));

    const auto sizes = decodeInt64Pair(buffers[1]);
    EXPECT_EQ(sizes[0], 320);
    EXPECT_EQ(sizes[1], 544);
}

TEST(ObjectDetectionBatchTest, EdgeCrafterPreprocessUsesOriginalImageSizeMetadata) {
    auto task = TaskFactory::createTaskInstance("edgecrafter", multiInputDetectionModelInfo());
    ASSERT_NE(task, nullptr);

    const auto buffers = task->preprocess({cv::Mat::zeros(123, 456, CV_8UC3)});

    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_FALSE(buffers[0].empty());
    ASSERT_EQ(buffers[1].size(), 2u * sizeof(int64_t));

    const auto sizes = decodeInt64Pair(buffers[1]);
    EXPECT_EQ(sizes[0], 456);
    EXPECT_EQ(sizes[1], 123);
}
