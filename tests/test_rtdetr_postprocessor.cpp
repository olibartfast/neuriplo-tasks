#include <gtest/gtest.h>
#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include "vision-core/object_detection/object_detection_task.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <variant>

using namespace vision_core;

class RtDetrPostprocessorTest : public ::testing::Test {
protected:
    void createMockRtDetrOutput(
        std::vector<TensorElement>& boxes, 
        std::vector<TensorElement>& scores,
        int num_queries, int num_classes) {
        
        // Boxes: [1, num_queries, 4] (cx, cy, w, h) normalized
        boxes.resize(num_queries * 4, 0.0f);
        // Scores: [1, num_queries, num_classes] (logits or probs)
        scores.resize(num_queries * num_classes, 0.0f);
        
        // Add detection at index 0
        boxes[0] = 0.5f; // cx
        boxes[1] = 0.5f; // cy
        boxes[2] = 0.2f; // w
        boxes[3] = 0.2f; // h
        
        // High score for class 0
        scores[0 * num_classes + 0] = 0.9f;
    }
};

TEST_F(RtDetrPostprocessorTest, StandardRtDetr) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_STYLE, cv::Size(640, 640), 0.5f);
    
    int num_queries = 300;
    int num_classes = 80;
    std::vector<TensorElement> boxes, scores;
    createMockRtDetrOutput(boxes, scores, num_queries, num_classes);
    
    std::vector<std::vector<TensorElement>> results = {boxes, scores};
    std::vector<std::vector<int64_t>> shapes = {
        {1, num_queries, 4},
        {1, num_queries, num_classes}
    };
    cv::Size frame_size(640, 640);
    
    auto detections = processor.postprocess(results, shapes, frame_size);
    
    ASSERT_FALSE(detections.empty());
    EXPECT_EQ(detections[0].class_id, 0);
    EXPECT_GT(detections[0].class_confidence, 0.5f);
    
    // Check box scaling
    EXPECT_NEAR(detections[0].bbox.x, (0.5 - 0.1) * 640, 1.0);
    EXPECT_NEAR(detections[0].bbox.y, (0.5 - 0.1) * 640, 1.0);
}

TEST_F(RtDetrPostprocessorTest, UltralyticsRtDetr) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_UL, cv::Size(640, 640), 0.5f);
    
    // Ultralytics RT-DETR output is similar to YOLO: [1, 4+num_classes, num_queries]
    // But our implementation expects 2 tensors for all RT-DETR types currently
    // So we should mock 2 tensors even if UL might export differently in some versions
    
    int num_queries = 300;
    int num_classes = 80;
    std::vector<TensorElement> boxes, scores;
    createMockRtDetrOutput(boxes, scores, num_queries, num_classes);
    
    std::vector<std::vector<TensorElement>> results = {boxes, scores};
    std::vector<std::vector<int64_t>> shapes = {
        {1, num_queries, 4},
        {1, num_queries, num_classes}
    };
    cv::Size frame_size(640, 640);
    
    auto detections = processor.postprocess(results, shapes, frame_size);
    
    ASSERT_FALSE(detections.empty());
    EXPECT_EQ(detections[0].class_id, 0);
    EXPECT_GT(detections[0].class_confidence, 0.5f);
}

TEST_F(RtDetrPostprocessorTest, EmptyInput) {
    RtDetrPostprocessor processor(ObjectDetectionTask::ModelType::RT_DETR_STYLE, cv::Size(640, 640), 0.5f);
    
    // Should throw because RT-DETR requires 2 tensors
    EXPECT_THROW({
        processor.postprocess({}, {}, cv::Size(640, 640));
    }, std::runtime_error);
}
