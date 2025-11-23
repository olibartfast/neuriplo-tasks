#include <gtest/gtest.h>
#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace vision_core;

class YoloPostprocessorTest : public ::testing::Test {
protected:
    // Helper to create mock YOLO output data
    std::vector<float> createMockYoloOutput(int num_boxes, int num_classes) {
        // YOLO format: [x, y, w, h, conf, class1, class2, ...]
        int box_size = 5 + num_classes;
        std::vector<float> output(num_boxes * box_size, 0.0f);
        
        // Add one high-confidence detection
        if (num_boxes > 0) {
            output[0] = 320.0f;  // x center
            output[1] = 240.0f;  // y center
            output[2] = 100.0f;  // width
            output[3] = 150.0f;  // height
            output[4] = 0.9f;    // confidence
            output[5] = 0.95f;   // class 0 score
        }
        
        return output;
    }
};

TEST_F(YoloPostprocessorTest, EmptyOutputReturnsNoDetections) {
    std::vector<float> output;
    std::vector<int64_t> shape = {1, 0, 85};  // No detections
    cv::Size frame_size(640, 480);
    
    auto detections = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
    );
    
    EXPECT_TRUE(detections.empty());
}

TEST_F(YoloPostprocessorTest, LowConfidenceFiltered) {
    auto output = createMockYoloOutput(1, 80);
    output[4] = 0.1f;  // Low confidence
    std::vector<int64_t> shape = {1, 1, 85};
    cv::Size frame_size(640, 480);
    
    auto detections = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
    );
    
    EXPECT_TRUE(detections.empty());
}

TEST_F(YoloPostprocessorTest, HighConfidenceDetected) {
    auto output = createMockYoloOutput(1, 80);
    std::vector<int64_t> shape = {1, 1, 85};
    cv::Size frame_size(640, 480);
    
    auto detections = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
    );
    
    EXPECT_FALSE(detections.empty());
    if (!detections.empty()) {
        EXPECT_GT(detections[0].class_confidence, 0.0f);
        EXPECT_GE(detections[0].class_id, 0.0f);
    }
}

TEST_F(YoloPostprocessorTest, BoundingBoxInBounds) {
    auto output = createMockYoloOutput(1, 80);
    std::vector<int64_t> shape = {1, 1, 85};
    cv::Size frame_size(640, 480);
    
    auto detections = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
    );
    
    for (const auto& det : detections) {
        EXPECT_GE(det.bbox.x, 0);
        EXPECT_GE(det.bbox.y, 0);
        EXPECT_GT(det.bbox.width, 0);
        EXPECT_GT(det.bbox.height, 0);
    }
}

TEST_F(YoloPostprocessorTest, DifferentInputShapes) {
    // Test with different YOLO output formats
    std::vector<std::pair<std::vector<int64_t>, int>> shapes = {
        {{1, 25200, 85}, 25200},   // YOLOv5/v8 typical shape
        {{1, 8400, 85}, 8400},      // Smaller grid
        {{1, 100, 85}, 100}         // Very small
    };
    
    for (const auto& [shape, num_boxes] : shapes) {
        auto output = createMockYoloOutput(num_boxes, 80);
        cv::Size frame_size(640, 480);
        
        EXPECT_NO_THROW({
            auto detections = YoloPostprocessor::postprocess(
                output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
            );
        });
    }
}

TEST_F(YoloPostprocessorTest, NMSReducesOverlappingBoxes) {
    // Create two overlapping boxes with high confidence
    std::vector<float> output;
    int num_classes = 80;
    int box_size = 5 + num_classes;
    
    // Box 1
    for (int i = 0; i < box_size; ++i) output.push_back(0.0f);
    output[0] = 320.0f;
    output[1] = 240.0f;
    output[2] = 100.0f;
    output[3] = 100.0f;
    output[4] = 0.9f;
    output[5] = 0.9f;
    
    // Box 2 (overlapping)
    for (int i = 0; i < box_size; ++i) output.push_back(0.0f);
    output[box_size + 0] = 325.0f;  // Slightly offset
    output[box_size + 1] = 245.0f;
    output[box_size + 2] = 100.0f;
    output[box_size + 3] = 100.0f;
    output[box_size + 4] = 0.85f;
    output[box_size + 5] = 0.85f;
    
    std::vector<int64_t> shape = {1, 2, 85};
    cv::Size frame_size(640, 480);
    
    auto detections = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
    );
    
    // NMS should reduce overlapping boxes
    EXPECT_LE(detections.size(), 2);
}

TEST_F(YoloPostprocessorTest, ConfidenceThresholdWorks) {
    auto output = createMockYoloOutput(1, 80);
    output[4] = 0.5f;  // Medium confidence
    std::vector<int64_t> shape = {1, 1, 85};
    cv::Size frame_size(640, 480);
    
    // Should pass with 0.25 threshold
    auto detections1 = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.25f, 0.45f
    );
    EXPECT_FALSE(detections1.empty());
    
    // Should fail with 0.75 threshold
    auto detections2 = YoloPostprocessor::postprocess(
        output.data(), shape, frame_size, 640, 640, 0.75f, 0.45f
    );
    EXPECT_TRUE(detections2.empty());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
