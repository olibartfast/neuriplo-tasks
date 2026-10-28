#include "neuriplo/tasks/core/batch_postprocess.hpp"
#include "neuriplo/tasks/core/batch_preprocess.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/object_detection/yolo_postprocessor.hpp"
#include "vision_test_utils.hpp"

#include <array>
#include <cstring>
#include <gtest/gtest.h>

using namespace neuriplo_tasks;

namespace {

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

ModelInfo multiInputDetectionModelInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 320, 544}, {1, 2}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"images", "orig_target_sizes"};
    info.input_datatypes = {"FP32", "INT64"};
    info.output_names = {"scores", "boxes"};
    info.input_types = {neuriplo_tasks::PixelType::Float32, neuriplo_tasks::PixelType::Int32};
    info.batch_size_ = 1;
    info.max_batch_size_ = 1;
    return info;
}

ModelInfo genericMultiInputDetectionModelInfo() {
    ModelInfo info = multiInputDetectionModelInfo();
    info.input_names = {"input_0", "input_1"};
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
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    const int anchors = 8;
    const int num_classes = 4;
    const Tensor tensor = makeBatchedYoloTensor(2, anchors, num_classes);

    const auto detections = processor.postprocess({tensor}, neuriplo_tasks::Size(640, 640));

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
    request.images = {neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(150, 200, 3, 0)};

    const auto pre = batchPreprocess(*task, request);
    EXPECT_EQ(pre.batch_size, 2);
    EXPECT_EQ(pre.buffers.size(), 2u);

    const Tensor tensor = makeBatchedYoloTensor(2, 8, 4);
    const auto direct = task->postprocess(neuriplo_tasks::Size(640, 640), {tensor});
    const auto post = batchPostprocess(*task, neuriplo_tasks::Size(640, 640), {tensor}, pre.batch_size);

    EXPECT_EQ(post.batch_size, 2);
    EXPECT_FALSE(direct.empty());
    EXPECT_EQ(post.results.size(), direct.size());
}

TEST(ObjectDetectionBatchTest, RtDetrPreprocessUsesModelInputSizeMetadata) {
    auto task = TaskFactory::createTaskInstance("rtdetr", multiInputDetectionModelInfo());
    ASSERT_NE(task, nullptr);

    const auto buffers = task->preprocess({neuriplo_tasks::vision_test::makeImage(456, 123, 3, 0)});

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

    const auto buffers = task->preprocess({neuriplo_tasks::vision_test::makeImage(456, 123, 3, 0)});

    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_FALSE(buffers[0].empty());
    ASSERT_EQ(buffers[1].size(), 2u * sizeof(int64_t));

    const auto sizes = decodeInt64Pair(buffers[1]);
    EXPECT_EQ(sizes[0], 456);
    EXPECT_EQ(sizes[1], 123);
}

TEST(ObjectDetectionBatchTest, EdgeCrafterPreprocessUsesShapeWhenSizeInputNameIsGeneric) {
    auto task = TaskFactory::createTaskInstance("edgecrafter", genericMultiInputDetectionModelInfo());
    ASSERT_NE(task, nullptr);

    const auto buffers = task->preprocess({neuriplo_tasks::vision_test::makeImage(456, 123, 3, 0)});

    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_FALSE(buffers[0].empty());
    ASSERT_EQ(buffers[1].size(), 2u * sizeof(int64_t));

    const auto sizes = decodeInt64Pair(buffers[1]);
    EXPECT_EQ(sizes[0], 456);
    EXPECT_EQ(sizes[1], 123);
}

TEST(ObjectDetectionBatchTest, DetectorSizeBuffersFollowEachRuntimeDatatype) {
    for (const char* model : {"rtdetr", "rtdetrv2", "dfine", "deim", "deimv2", "ecdet"}) {
        for (const std::string datatype : {"INT32", "INT64", "FP32"}) {
            SCOPED_TRACE(std::string(model) + " " + datatype);
            auto info = multiInputDetectionModelInfo();
            info.input_datatypes = {"FP32", datatype};
            auto task = TaskFactory::createTaskInstance(model, info);
            const auto buffers = task->preprocess({neuriplo_tasks::vision_test::makeImage(456, 123, 3, 127)});
            ASSERT_EQ(buffers.size(), 2U);
            ASSERT_EQ(buffers[0].size(), 3U * 320U * 544U * sizeof(float));
            float first_pixel = 0.0f;
            std::memcpy(&first_pixel, buffers[0].data(), sizeof(first_pixel));
            const bool original_size = std::string(model) == "ecdet";
            const float expected_pixel = original_size ? (127.0f / 255.0f - 0.485f) / 0.229f : 127.0f / 255.0f;
            EXPECT_NEAR(first_pixel, expected_pixel, 1e-5f);
            const std::array<int64_t, 2> expected =
                original_size ? std::array<int64_t, 2>{456, 123} : std::array<int64_t, 2>{320, 544};
            if (datatype == "INT64") {
                ASSERT_EQ(buffers[1].size(), 2U * sizeof(int64_t));
                EXPECT_EQ(decodeInt64Pair(buffers[1]), expected);
            } else if (datatype == "INT32") {
                ASSERT_EQ(buffers[1].size(), 2U * sizeof(int32_t));
                std::array<int32_t, 2> values{};
                std::memcpy(values.data(), buffers[1].data(), buffers[1].size());
                EXPECT_EQ(values[0], expected[0]);
                EXPECT_EQ(values[1], expected[1]);
            } else {
                ASSERT_EQ(buffers[1].size(), 2U * sizeof(float));
                std::array<float, 2> values{};
                std::memcpy(values.data(), buffers[1].data(), buffers[1].size());
                EXPECT_FLOAT_EQ(values[0], static_cast<float>(expected[0]));
                EXPECT_FLOAT_EQ(values[1], static_cast<float>(expected[1]));
            }
        }
    }
}

TEST(ObjectDetectionBatchTest, RejectsMissingOrUnsupportedRuntimeSizeTypes) {
    for (const std::vector<std::string>& types :
         {std::vector<std::string>{"FP32"}, {"FP32", ""}, {"FP32", "BOOL"}, {"FP32", "UINT8"}, {"FP32", "FP16"}}) {
        auto info = multiInputDetectionModelInfo();
        info.input_datatypes = types;
        if (types.size() != 2) {
            EXPECT_THROW(TaskFactory::createTaskInstance("deim", info), std::invalid_argument);
        } else {
            auto task = TaskFactory::createTaskInstance("deim", info);
            EXPECT_THROW(task->preprocess({neuriplo_tasks::vision_test::makeImage(456, 123, 3, 127)}),
                         std::invalid_argument);
        }
    }
}

TEST(ObjectDetectionBatchTest, RuntimeSizeTypePreservesBatchedOriginalDimensions) {
    auto info = multiInputDetectionModelInfo();
    info.input_shapes = {{2, 3, 320, 544}, {2, 2}};
    info.input_datatypes = {"FP32", "INT32"};
    info.batch_size_ = 2;
    info.max_batch_size_ = 2;
    auto task = TaskFactory::createTaskInstance("ecdet", info);
    const auto buffers = task->preprocess({neuriplo_tasks::vision_test::makeImage(456, 123, 3, 0),
                                           neuriplo_tasks::vision_test::makeImage(210, 310, 3, 255)});
    ASSERT_EQ(buffers.size(), 2U);
    ASSERT_EQ(buffers[0].size(), 2U * 3U * 320U * 544U * sizeof(float));
    ASSERT_EQ(buffers[1].size(), 4U * sizeof(int32_t));
    std::array<int32_t, 4> sizes{};
    std::memcpy(sizes.data(), buffers[1].data(), buffers[1].size());
    EXPECT_EQ(sizes, (std::array<int32_t, 4>{456, 123, 210, 310}));
}
