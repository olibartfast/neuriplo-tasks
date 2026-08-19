#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/object_detection/yolo_postprocessor.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>
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
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    std::vector<Tensor> tensors = {};
    neuriplo_tasks::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_TRUE(detections.empty());
}

TEST_F(YoloPostprocessorTest, HighConfidenceDetected) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    int num_boxes = 8400;
    int num_classes = 80;
    auto output = createMockYoloOutput(num_boxes, num_classes);

    std::vector<Tensor> tensors = {Tensor(output, {1, 4 + num_classes, num_boxes})};
    neuriplo_tasks::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_FALSE(detections.empty());
    if (!detections.empty()) {
        EXPECT_GT(detections[0].class_confidence, 0.9f);
        EXPECT_EQ(detections[0].class_id, 0);
    }
}

TEST_F(YoloPostprocessorTest, LowConfidenceFiltered) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_STANDARD, neuriplo_tasks::Size(640, 640), 0.99f,
                                0.45f); // Very high threshold

    int num_boxes = 8400;
    int num_classes = 80;
    auto output = createMockYoloOutput(num_boxes, num_classes);

    std::vector<Tensor> tensors = {Tensor(output, {1, 4 + num_classes, num_boxes})};
    neuriplo_tasks::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    EXPECT_TRUE(detections.empty());
}

TEST_F(YoloPostprocessorTest, YoloNmsFreeFormat) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NMS_FREE, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

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
    neuriplo_tasks::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}

// Ultralytics YOLOv10 / YOLO26 exports emit xyxy in input-image pixels, not in
// normalized [0, 1]. Scaling those by the input size again put every box roughly
// 640x off-frame, so the model appeared to detect nothing at all.
TEST_F(YoloPostprocessorTest, YoloNmsFreeAcceptsPixelCoordinates) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NMS_FREE, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    const int num_dets = 300;
    const int dims = 6;
    std::vector<TensorElement> output(num_dets * dims, 0.0f);

    // Same box as the normalized test, expressed in 640x640 input pixels.
    output[0] = 160.0f; // x1
    output[1] = 160.0f; // y1
    output[2] = 320.0f; // x2
    output[3] = 320.0f; // y2
    output[4] = 0.9f;   // score
    output[5] = 1.0f;   // class 1

    std::vector<Tensor> tensors = {Tensor(output, {1, num_dets, dims})};
    neuriplo_tasks::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 1);
    // Identical geometry to YoloNmsFreeFormat, which feeds the normalized form.
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}

// A pixel-coordinate box that fills the frame must not be mistaken for a
// normalized one, and vice versa: the decision is made per tensor.
TEST_F(YoloPostprocessorTest, YoloNmsFreeKeepsConventionsSeparate) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NMS_FREE, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    const int num_dets = 8;
    const int dims = 6;

    std::vector<TensorElement> pixels(num_dets * dims, 0.0f);
    pixels[0] = 0.0f;
    pixels[1] = 0.0f;
    pixels[2] = 640.0f;
    pixels[3] = 640.0f;
    pixels[4] = 0.9f;
    pixels[5] = 0.0f;
    auto from_pixels = processor.postprocess({Tensor(pixels, {1, num_dets, dims})}, neuriplo_tasks::Size(640, 640));

    std::vector<TensorElement> normalized(num_dets * dims, 0.0f);
    normalized[0] = 0.0f;
    normalized[1] = 0.0f;
    normalized[2] = 1.0f;
    normalized[3] = 1.0f;
    normalized[4] = 0.9f;
    normalized[5] = 0.0f;
    auto from_normalized =
        processor.postprocess({Tensor(normalized, {1, num_dets, dims})}, neuriplo_tasks::Size(640, 640));

    ASSERT_EQ(from_pixels.size(), 1);
    ASSERT_EQ(from_normalized.size(), 1);
    EXPECT_EQ(from_pixels[0].bbox, from_normalized[0].bbox);
    EXPECT_EQ(from_pixels[0].bbox, BoundingBox(0, 0, 640, 640));
}

TEST_F(YoloPostprocessorTest, YoloNmsFreeAppliesNmsPerClass) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NMS_FREE, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    const int num_dets = 300;
    const int dims = 6;
    std::vector<TensorElement> output(num_dets * dims, 0.0f);

    // Overlapping class-1 detections should collapse to the higher score.
    output[0] = 0.25f;
    output[1] = 0.25f;
    output[2] = 0.50f;
    output[3] = 0.50f;
    output[4] = 0.90f;
    output[5] = 1.0f;

    output[6] = 0.26f;
    output[7] = 0.26f;
    output[8] = 0.51f;
    output[9] = 0.51f;
    output[10] = 0.80f;
    output[11] = 1.0f;

    // Same box, different class: class-aware NMS should keep it.
    output[12] = 0.25f;
    output[13] = 0.25f;
    output[14] = 0.50f;
    output[15] = 0.50f;
    output[16] = 0.70f;
    output[17] = 2.0f;

    auto detections = processor.postprocess({Tensor(output, {1, num_dets, dims})}, neuriplo_tasks::Size(640, 480));

    ASSERT_EQ(detections.size(), 2);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.9f);
    EXPECT_EQ(detections[1].class_id, 2);
    EXPECT_FLOAT_EQ(detections[1].class_confidence, 0.7f);
}

TEST_F(YoloPostprocessorTest, YoloNasFormatScalesXyxyFromModelSpace) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NAS, neuriplo_tasks::Size(640, 640), 0.25f, 0.45f);

    std::vector<TensorElement> boxes = {160.0f, 160.0f, 320.0f, 320.0f, 10.0f, 10.0f, 20.0f, 20.0f};
    std::vector<TensorElement> scores = {0.1f, 0.8f, 0.2f, 0.1f, 0.1f, 0.1f};

    std::vector<Tensor> tensors = {Tensor(boxes, {1, 2, 4}), Tensor(scores, {1, 2, 3})};
    neuriplo_tasks::Size frame_size(640, 480);

    auto detections = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.8f);
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}

TEST_F(YoloPostprocessorTest, YoloNasKeepsUnclampedBoxCrossingLetterboxPadding) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_NAS, neuriplo_tasks::Size(640, 640), 0.25f, 0.45f);

    std::vector<TensorElement> boxes = {0.0f, 0.0f, 100.0f, 100.0f};
    std::vector<TensorElement> scores = {0.9f};

    auto detections =
        processor.postprocess({Tensor(boxes, {1, 1, 4}), Tensor(scores, {1, 1, 1})}, neuriplo_tasks::Size(640, 480));

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].bbox, BoundingBox(0, -80, 100, 100));
}

TEST_F(YoloPostprocessorTest, YoloV7E2EFormatScalesXyxyFromModelSpace) {
    YoloPostprocessor processor(ObjectDetectionTask::ModelType::YOLO_V7_E2E, neuriplo_tasks::Size(640, 640), 0.25f,
                                0.45f);

    Tensor num_dets({1.0f}, {1, 1});
    Tensor boxes({160.0f, 160.0f, 320.0f, 320.0f}, {1, 1, 4});
    Tensor scores({0.85f}, {1, 1});
    Tensor classes({2.0f}, {1, 1});

    auto detections = processor.postprocess({num_dets, boxes, scores, classes}, neuriplo_tasks::Size(640, 480));

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_id, 2);
    EXPECT_FLOAT_EQ(detections[0].class_confidence, 0.85f);
    EXPECT_EQ(detections[0].bbox, BoundingBox(160, 80, 160, 160));
}
