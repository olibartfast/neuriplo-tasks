#include "vision-core/core/result_types.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vision_core;

class ResultTypesTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ResultTypesTest, ClassificationDefaultConstruction) {
    Classification cls;
    EXPECT_FLOAT_EQ(cls.class_id, -1.0f);
    EXPECT_FLOAT_EQ(cls.class_confidence, 0.0f);
}

TEST_F(ResultTypesTest, ClassificationParameterizedConstruction) {
    Classification cls(5.0f, 0.95f);
    EXPECT_FLOAT_EQ(cls.class_id, 5.0f);
    EXPECT_FLOAT_EQ(cls.class_confidence, 0.95f);
}

TEST_F(ResultTypesTest, DetectionDefaultConstruction) {
    Detection det;
    EXPECT_FLOAT_EQ(det.class_id, -1.0f);
    EXPECT_FLOAT_EQ(det.class_confidence, 0.0f);
    EXPECT_EQ(det.bbox.x, 0);
    EXPECT_EQ(det.bbox.y, 0);
    EXPECT_EQ(det.bbox.width, 0);
    EXPECT_EQ(det.bbox.height, 0);
}

TEST_F(ResultTypesTest, DetectionParameterizedConstruction) {
    cv::Rect bbox(100, 200, 50, 75);
    Detection det(bbox, 0.85f, 3);

    EXPECT_FLOAT_EQ(det.class_id, 3.0f);
    EXPECT_FLOAT_EQ(det.class_confidence, 0.85f);
    EXPECT_EQ(det.bbox.x, 100);
    EXPECT_EQ(det.bbox.y, 200);
    EXPECT_EQ(det.bbox.width, 50);
    EXPECT_EQ(det.bbox.height, 75);
}

TEST_F(ResultTypesTest, DetectionInheritsFromClassification) {
    Detection det(cv::Rect(10, 20, 30, 40), 0.9f, 7);

    // Can use as Classification
    Classification& cls_ref = det;
    EXPECT_FLOAT_EQ(cls_ref.class_id, 7.0f);
    EXPECT_FLOAT_EQ(cls_ref.class_confidence, 0.9f);
}

TEST_F(ResultTypesTest, InstanceSegmentationDefaultConstruction) {
    InstanceSegmentation seg;
    EXPECT_FLOAT_EQ(seg.class_id, -1.0f);
    EXPECT_FLOAT_EQ(seg.class_confidence, 0.0f);
    EXPECT_TRUE(seg.mask_data.empty());
    EXPECT_EQ(seg.mask_height, 0);
    EXPECT_EQ(seg.mask_width, 0);
}

TEST_F(ResultTypesTest, InstanceSegmentationParameterizedConstruction) {
    cv::Rect bbox(50, 60, 100, 120);
    InstanceSegmentation seg(bbox, 0.92f, 1);

    EXPECT_FLOAT_EQ(seg.class_id, 1.0f);
    EXPECT_FLOAT_EQ(seg.class_confidence, 0.92f);
    EXPECT_EQ(seg.bbox.x, 50);
    EXPECT_EQ(seg.bbox.y, 60);
}

TEST_F(ResultTypesTest, InstanceSegmentationInheritsFromDetection) {
    InstanceSegmentation seg(cv::Rect(1, 2, 3, 4), 0.8f, 2);

    // Can use as Detection
    Detection& det_ref = seg;
    EXPECT_EQ(det_ref.bbox.x, 1);

    // Can use as Classification
    Classification& cls_ref = seg;
    EXPECT_FLOAT_EQ(cls_ref.class_confidence, 0.8f);
}

TEST_F(ResultTypesTest, InstanceSegmentationMaskData) {
    InstanceSegmentation seg;
    seg.mask_data = {255, 0, 128, 64};
    seg.mask_width = 2;
    seg.mask_height = 2;

    EXPECT_EQ(seg.mask_data.size(), 4);
    EXPECT_EQ(seg.mask_data[0], 255);
    EXPECT_EQ(seg.mask_data[2], 128);
    EXPECT_EQ(seg.mask_width, 2);
    EXPECT_EQ(seg.mask_height, 2);
}

TEST_F(ResultTypesTest, OpticalFlowDefaultConstruction) {
    OpticalFlow flow;
    EXPECT_TRUE(flow.flow.empty());
    EXPECT_TRUE(flow.raw_flow.empty());
    EXPECT_FLOAT_EQ(flow.max_displacement, 0.0f);
}

TEST_F(ResultTypesTest, OpticalFlowWithData) {
    OpticalFlow flow;
    flow.flow = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    flow.raw_flow = cv::Mat(100, 100, CV_32FC2);
    flow.max_displacement = 15.5f;

    EXPECT_EQ(flow.flow.rows, 100);
    EXPECT_EQ(flow.flow.cols, 100);
    EXPECT_EQ(flow.raw_flow.type(), CV_32FC2);
    EXPECT_FLOAT_EQ(flow.max_displacement, 15.5f);
}

TEST_F(ResultTypesTest, VideoClassificationDefaultConstruction) {
    VideoClassification vid;
    EXPECT_FLOAT_EQ(vid.class_id, -1.0f);
    EXPECT_FLOAT_EQ(vid.class_confidence, 0.0f);
    EXPECT_TRUE(vid.action_label.empty());
    EXPECT_TRUE(vid.frame_scores.empty());
}

TEST_F(ResultTypesTest, VideoClassificationWithData) {
    VideoClassification vid;
    vid.class_id = 10.0f;
    vid.class_confidence = 0.88f;
    vid.action_label = "running";
    vid.frame_scores = {0.8f, 0.85f, 0.9f, 0.88f};

    EXPECT_FLOAT_EQ(vid.class_id, 10.0f);
    EXPECT_EQ(vid.action_label, "running");
    EXPECT_EQ(vid.frame_scores.size(), 4);
}

TEST_F(ResultTypesTest, ResultVariantHoldsClassification) {
    Classification cls(3.0f, 0.75f);
    Result result = cls;

    EXPECT_TRUE(std::holds_alternative<Classification>(result));
    EXPECT_FALSE(std::holds_alternative<Detection>(result));

    auto& stored_cls = std::get<Classification>(result);
    EXPECT_FLOAT_EQ(stored_cls.class_id, 3.0f);
}

TEST_F(ResultTypesTest, ResultVariantHoldsDetection) {
    Detection det(cv::Rect(1, 2, 3, 4), 0.9f, 5);
    Result result = det;

    EXPECT_TRUE(std::holds_alternative<Detection>(result));
    EXPECT_FALSE(std::holds_alternative<Classification>(result));
}

TEST_F(ResultTypesTest, ResultVariantHoldsInstanceSegmentation) {
    InstanceSegmentation seg(cv::Rect(10, 20, 30, 40), 0.95f, 2);
    Result result = seg;

    EXPECT_TRUE(std::holds_alternative<InstanceSegmentation>(result));
}

TEST_F(ResultTypesTest, ResultVariantHoldsOpticalFlow) {
    OpticalFlow flow;
    flow.max_displacement = 10.0f;
    Result result = flow;

    EXPECT_TRUE(std::holds_alternative<OpticalFlow>(result));
    auto& stored_flow = std::get<OpticalFlow>(result);

    EXPECT_FLOAT_EQ(stored_flow.max_displacement, 10.0f);
}

TEST_F(ResultTypesTest, ResultVariantHoldsVideoClassification) {
    VideoClassification vid;
    vid.action_label = "jumping";
    Result result = vid;

    EXPECT_TRUE(std::holds_alternative<VideoClassification>(result));
}

TEST_F(ResultTypesTest, DepthEstimationDefaultConstruction) {
    DepthEstimation depth;
    EXPECT_TRUE(depth.depth.empty());
    EXPECT_TRUE(depth.normalized_depth.empty());
    EXPECT_FLOAT_EQ(depth.min_depth, 0.0f);
    EXPECT_FLOAT_EQ(depth.max_depth, 0.0f);
}

TEST_F(ResultTypesTest, ResultVariantHoldsDepthEstimation) {
    DepthEstimation depth;
    depth.depth = cv::Mat::ones(10, 10, CV_32FC1);
    Result result = depth;

    EXPECT_TRUE(std::holds_alternative<DepthEstimation>(result));
}

TEST_F(ResultTypesTest, TaskTypeEnumValues) {
    EXPECT_EQ(static_cast<int>(TaskType::OpticalFlow), 0);
    EXPECT_EQ(static_cast<int>(TaskType::Classification), 1);
    EXPECT_EQ(static_cast<int>(TaskType::Detection), 2);
    EXPECT_EQ(static_cast<int>(TaskType::InstanceSegmentation), 3);
    EXPECT_EQ(static_cast<int>(TaskType::VideoClassification), 4);
    EXPECT_EQ(static_cast<int>(TaskType::PoseEstimation), 5);
    EXPECT_EQ(static_cast<int>(TaskType::DepthEstimation), 6);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
