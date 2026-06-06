#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/object_detection/yolo_postprocessor.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <variant>
#include <vector>

using namespace neuriplo_tasks;

class YoloPostprocessorTest : public ::testing::Test {
  protected:
    // Helper to create mock YOLO output data
    std::vector<TensorElement> createMockYoloOutput(int num_boxes, int num_classes) {
        // YOLO format: [x, y, w, h, conf, class1, class2, ...]
        // Transposed: [batch, channels, anchors] -> [1, 5+num_classes, num_boxes]
        // Or [1, num_boxes, 5+num_classes] depending on model

        // Let's assume standard YOLOv8 format: [1, 4+num_classes, num_boxes]
        // where 4 is bbox, and rest are class scores. No objectness score in v8.

        int channels = 4 + num_classes;
        std::vector<TensorElement> output;
        output.reserve(channels * num_boxes);

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < num_boxes; ++i) {
                float val = 0.0f;
                // Add one high-confidence detection at index 0
                if (i == 0) {
                    if (c == 0)
                        val = 320.0f; // x
                    else if (c == 1)
                        val = 240.0f; // y
                    else if (c == 2)
                        val = 100.0f; // w
                    else if (c == 3)
                        val = 150.0f; // h
                    else if (c == 4)
                        val = 0.95f; // class 0 score
                }
                output.push_back(val);
            }
        }

        return output;
    }
};

TEST_F(YoloPostprocessorTest, EmptyOutputReturnsNoDetections) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, cv::Size(640, 640), 0.25f, 0.45f);

    std::vector<Tensor> tensors = {};
    cv::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_TRUE(detections.empty());
}

TEST_F(YoloPostprocessorTest, HighConfidenceDetected) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, cv::Size(640, 640), 0.25f, 0.45f);

    int num_boxes = 8400;
    int num_classes = 80;
    auto output = createMockYoloOutput(num_boxes, num_classes);

    std::vector<Tensor> tensors = {Tensor(output, {1, 4 + num_classes, num_boxes})};
    cv::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_FALSE(detections.empty());
    if (!detections.empty()) {
        EXPECT_GT(detections[0].class_confidence, 0.9f);
        EXPECT_EQ(detections[0].class_id, 0);
    }
}

TEST_F(YoloPostprocessorTest, LowConfidenceFiltered) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, cv::Size(640, 640), 0.99f,
                                0.45f); // Very high threshold

    int num_boxes = 8400;
    int num_classes = 80;
    auto output = createMockYoloOutput(num_boxes, num_classes);

    std::vector<Tensor> tensors = {Tensor(output, {1, 4 + num_classes, num_boxes})};
    cv::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_TRUE(detections.empty());
}

TEST_F(YoloPostprocessorTest, YoloNmsFreeFormat) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NMS_FREE, cv::Size(640, 640), 0.25f, 0.45f);

    // YOLOv10/YOLO26 output: [1, 300, 6] (x1, y1, x2, y2, score, class)
    int num_dets = 300;
    int dims = 6;
    std::vector<TensorElement> output(num_dets * dims, 0.0f);

    // Add one detection
    output[0] = 0.25f; // x1
    output[1] = 0.25f; // y1
    output[2] = 0.5f;  // x2
    output[3] = 0.5f;  // y2
    output[4] = 0.9f;  // score
    output[5] = 1.0f;  // class 1

    std::vector<Tensor> tensors = {Tensor(output, {1, num_dets, dims})};
    cv::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}

TEST_F(YoloPostprocessorTest, YoloNasFormatScalesXyxyFromModelSpace) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NAS, cv::Size(640, 640), 0.25f, 0.45f);

    std::vector<TensorElement> boxes = {160.0f, 160.0f, 320.0f, 320.0f, 10.0f, 10.0f, 20.0f, 20.0f};
    std::vector<TensorElement> scores = {0.1f, 0.8f, 0.2f, 0.1f, 0.1f, 0.1f};

    std::vector<Tensor> tensors = {Tensor(boxes, {1, 2, 4}), Tensor(scores, {1, 2, 3})};
    cv::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.8f);
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}

TEST_F(YoloPostprocessorTest, YoloNasKeepsUnclampedBoxCrossingLetterboxPadding) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NAS, cv::Size(640, 640), 0.25f, 0.45f);

    std::vector<TensorElement> boxes = {0.0f, 0.0f, 100.0f, 100.0f};
    std::vector<TensorElement> scores = {0.9f};

    auto detections = processor.postprocess({Tensor(boxes, {1, 1, 4}), Tensor(scores, {1, 1, 1})}, cv::Size(640, 480));

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].bbox, BoundingBox(0, -80, 100, 100));
}

TEST_F(YoloPostprocessorTest, YoloV7E2EFormatScalesXyxyFromModelSpace) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_V7_E2E, cv::Size(640, 640), 0.25f, 0.45f);

    Tensor num_dets({1.0f}, {1, 1});
    Tensor boxes({160.0f, 160.0f, 320.0f, 320.0f}, {1, 1, 4});
    Tensor scores({0.85f}, {1, 1});
    Tensor classes({2.0f}, {1, 1});

    auto detections = processor.postprocess({num_dets, boxes, scores, classes}, cv::Size(640, 480));

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 2);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.85f);
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}
