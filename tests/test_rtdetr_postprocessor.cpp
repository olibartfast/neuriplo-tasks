#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/object_detection/rtdetr_postprocessor.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>
#include <variant>
#include <vector>

using namespace neuriplo_tasks;

class RtDetrPostprocessorTest : public ::testing::Test {
  protected:
    void createMockRtDetrOutput(std::vector<TensorElement>& scores, std::vector<TensorElement>& boxes,
                                std::vector<TensorElement>& labels, int num_queries) {

        // RT-DETR outputs:
        // scores: [1, 300] - confidence per detection
        // boxes: [1, 300, 4] - x1, y1, x2, y2 coordinates
        // labels: [1, 300] - class ID per detection

        scores.resize(num_queries, 0.0f);
        boxes.resize(num_queries * 4, 0.0f);
        labels.resize(num_queries, 0.0f);

        // Add high-confidence detection at index 0
        scores[0] = 0.9f; // high confidence

        // Box coordinates (x1, y1, x2, y2) normalized
        boxes[0] = 256.0f; // x1 (already in pixel coordinates)
        boxes[1] = 256.0f; // y1
        boxes[2] = 384.0f; // x2
        boxes[3] = 384.0f; // y2

        labels[0] = 0; // class 0 (as int)
    }

    void createMockUltralyticsOutput(std::vector<TensorElement>& output, int num_queries, int num_classes) {

        // Ultralytics RT-DETR output: [1, num_queries, 4+num_classes]
        output.resize(num_queries * (4 + num_classes), 0.0f);

        // Add detection at query 0
        int query = 0;
        int dims = 4 + num_classes;

        // Box coordinates (x1, y1, x2, y2) in pixels
        output[query * dims + 0] = 256.0f; // x1
        output[query * dims + 1] = 256.0f; // y1
        output[query * dims + 2] = 384.0f; // x2
        output[query * dims + 3] = 384.0f; // y2

        // High score for class 0
        output[query * dims + 4 + 0] = 0.9f;
    }
};

TEST_F(RtDetrPostprocessorTest, StandardRtDetr) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_STYLE, neuriplo_tasks::Size(640, 640), 0.5f);

    int num_queries = 300;
    std::vector<TensorElement> scores, boxes, labels;
    createMockRtDetrOutput(scores, boxes, labels, num_queries);

    std::vector<Tensor> tensors = {Tensor(scores, {1, num_queries}), Tensor(boxes, {1, num_queries, 4}),
                                   Tensor(labels, {1, num_queries})};
    neuriplo_tasks::Size frame_size(640, 640);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_FALSE(detections.empty()) << "No detections found, size: " << detections.size();
    EXPECT_EQ(detections[0].class_id, 0);
    EXPECT_GT(detections[0].class_confidence, 0.5f);

    // Check box scaling - box from (256, 256) to (384, 384) pixels
    EXPECT_NEAR(detections[0].bbox.x, 256, 1.0);            // x1 = 256
    EXPECT_NEAR(detections[0].bbox.y, 256, 1.0);            // y1 = 256
    EXPECT_NEAR(detections[0].bbox.width, 384 - 256, 1.0);  // width = 128
    EXPECT_NEAR(detections[0].bbox.height, 384 - 256, 1.0); // height = 128
}

TEST_F(RtDetrPostprocessorTest, UltralyticsRtDetr) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_UL, neuriplo_tasks::Size(640, 640), 0.5f);

    // Ultralytics RT-DETR output: [1, 4+num_classes, num_queries]
    int num_queries = 300;
    int num_classes = 80;
    std::vector<TensorElement> output;
    createMockUltralyticsOutput(output, num_queries, num_classes);

    std::vector<Tensor> tensors = {Tensor(output, {1, num_queries, 4 + num_classes})};
    neuriplo_tasks::Size frame_size(640, 640);

    auto detections = processor.postprocess(tensors, frame_size);

    ASSERT_FALSE(detections.empty());
    EXPECT_EQ(detections[0].class_id, 0);
    EXPECT_GT(detections[0].class_confidence, 0.5f);
}

TEST_F(RtDetrPostprocessorTest, EmptyInput) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_STYLE, neuriplo_tasks::Size(640, 640), 0.5f);

    // Should throw because RT-DETR style requires 3 tensors
    EXPECT_THROW({ processor.postprocess({}, neuriplo_tasks::Size(640, 640)); }, std::runtime_error);
}

TEST_F(RtDetrPostprocessorTest, UltralyticsEndToEndNormalizedBoxes) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_UL, Size(640, 640), 0.5f);
    Tensor output({0.5f, 0.5f, 0.25f, 0.5f, 0.9f, 16.0f, 0.5f, 0.5f, 0.25f, 0.5f, 0.1f, 1.0f}, {1, 2, 6});
    const auto detections = processor.postprocess({output}, Size(800, 400));
    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].class_id, 16);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
    EXPECT_EQ(detections[0].bbox.x, 300);
    EXPECT_EQ(detections[0].bbox.y, 100);
    EXPECT_EQ(detections[0].bbox.width, 200);
    EXPECT_EQ(detections[0].bbox.height, 200);
}

TEST_F(RtDetrPostprocessorTest, UltralyticsEndToEndTruncatedTensor) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_UL, Size(640, 640), 0.5f);
    Tensor output({0.5f, 0.5f, 0.25f}, {1, 1, 6});
    EXPECT_TRUE(processor.postprocess({output}, Size(800, 400)).empty());
}

TEST_F(RtDetrPostprocessorTest, TorchScriptLabelsFirstLayoutWithoutNames) {
    // TorchScript exports keep the upstream tuple order (labels, boxes,
    // scores) and expose no output names. Regression for the DEIM adapter
    // rendering the label tensor as a confidence score.
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_STYLE, Size(640, 640), 0.25f);

    constexpr int kQueries = 3;
    std::vector<TensorElement> labels = {int64_t{16}, int64_t{1}, int64_t{0}};
    std::vector<TensorElement> boxes = {100.0f, 200.0f, 300.0f, 500.0f, 10.0f, 20.0f,
                                        30.0f,  40.0f,  0.0f,   0.0f,   0.0f,  0.0f};
    std::vector<TensorElement> scores = {0.86f, 0.81f, 0.10f};

    std::vector<Tensor> tensors = {Tensor(labels, {1, kQueries}), Tensor(boxes, {1, kQueries, 4}),
                                   Tensor(scores, {1, kQueries})};

    const auto detections = processor.postprocess(tensors, Size(640, 640));

    ASSERT_EQ(detections.size(), 2u);
    EXPECT_EQ(detections[0].class_id, 16);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.86f);
    EXPECT_EQ(detections[0].bbox.x, 100);
    EXPECT_EQ(detections[0].bbox.y, 200);
    EXPECT_EQ(detections[0].bbox.width, 200);
    EXPECT_EQ(detections[0].bbox.height, 300);
    EXPECT_EQ(detections[1].class_id, 1);
    EXPECT_FLOAT_EQ(detections[1].class_confidence, 0.81f);
}

TEST_F(RtDetrPostprocessorTest, NamedLayoutWinsOverShapeInference) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_STYLE, Size(640, 640), 0.25f,
                                  {"labels", "boxes", "scores"});

    std::vector<TensorElement> labels = {int64_t{7}};
    std::vector<TensorElement> boxes = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<TensorElement> scores = {0.9f};
    std::vector<Tensor> tensors = {Tensor(labels, {1, 1}), Tensor(boxes, {1, 1, 4}), Tensor(scores, {1, 1})};

    const auto detections = processor.postprocess(tensors, Size(640, 640));
    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].class_id, 7);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
}
