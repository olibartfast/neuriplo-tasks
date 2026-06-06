#include "neuriplo/tasks/core/bbox_processor.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace neuriplo_tasks;

class BboxProcessorTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(BboxProcessorTest, CalculateBoundingBoxXYWH) {
    cv::Size image_size(640, 480);
    std::vector<float> bbox = {320.0f, 240.0f, 100.0f, 80.0f}; // x_center, y_center, w, h in network space
    int network_width = 640;
    int network_height = 640;

    auto result = BBoxProcessor::calculate_bounding_box(image_size, bbox, network_width, network_height);

    // Verify the result is valid
    EXPECT_GE(result.x, 0);
    EXPECT_GE(result.y, 0);
    EXPECT_GT(result.width, 0);
    EXPECT_GT(result.height, 0);
}

TEST_F(BboxProcessorTest, CalculateBoundingBoxFromXYXY) {
    cv::Size image_size(640, 480);
    std::vector<float> bbox = {100.0f, 100.0f, 200.0f, 200.0f}; // x1, y1, x2, y2 in network space
    int network_width = 640;
    int network_height = 640;

    auto result = BBoxProcessor::calculate_bounding_box_from_xyxy(image_size, bbox, network_width, network_height);

    // Verify the result is valid
    EXPECT_GE(result.x, 0);
    EXPECT_GE(result.y, 0);
    EXPECT_GT(result.width, 0);
    EXPECT_GT(result.height, 0);
}

TEST_F(BboxProcessorTest, ScaleToOriginal) {
    cv::Rect bbox(100, 100, 50, 50); // bbox in 640x640 network coordinates
    cv::Size original_size(1920, 1080);
    cv::Size network_size(640, 640);

    auto scaled = BBoxProcessor::scale_to_original(bbox, original_size, network_size);

    // At 3x scaling (1920/640), the bbox should be scaled proportionally
    EXPECT_GT(scaled.width, bbox.width);
    EXPECT_GT(scaled.height, bbox.height);
}

TEST_F(BboxProcessorTest, ClampToBounds) {
    cv::Rect bbox(600, 450, 100, 100); // Partially outside 640x480 image
    cv::Size image_size(640, 480);

    BBoxProcessor::clamp_to_bounds(bbox, image_size);

    // Verify bbox is within image bounds
    EXPECT_GE(bbox.x, 0);
    EXPECT_GE(bbox.y, 0);
    EXPECT_LE(bbox.x + bbox.width, image_size.width);
    EXPECT_LE(bbox.y + bbox.height, image_size.height);
}

TEST_F(BboxProcessorTest, ClampCompletelyOutside) {
    cv::Rect bbox(700, 500, 100, 100); // Completely outside 640x480 image
    cv::Size image_size(640, 480);

    BBoxProcessor::clamp_to_bounds(bbox, image_size);

    // Verify bbox is clamped to image bounds
    EXPECT_GE(bbox.x, 0);
    EXPECT_GE(bbox.y, 0);
    EXPECT_LE(bbox.x, image_size.width);
    EXPECT_LE(bbox.y, image_size.height);
}

TEST_F(BboxProcessorTest, ZeroSizeBbox) {
    cv::Rect bbox(100, 100, 0, 0);
    cv::Size image_size(640, 480);

    BBoxProcessor::clamp_to_bounds(bbox, image_size);

    // Should handle zero-size boxes gracefully
    EXPECT_EQ(bbox.width, 0);
    EXPECT_EQ(bbox.height, 0);
}

TEST_F(BboxProcessorTest, NegativeCoordinates) {
    cv::Rect bbox(-50, -50, 100, 100); // Starts outside image
    cv::Size image_size(640, 480);

    BBoxProcessor::clamp_to_bounds(bbox, image_size);

    // Verify negative coordinates are clamped to 0
    EXPECT_GE(bbox.x, 0);
    EXPECT_GE(bbox.y, 0);
}

TEST_F(BboxProcessorTest, IdentityScaling) {
    cv::Rect bbox(100, 100, 50, 50);
    cv::Size size(640, 640);

    auto scaled = BBoxProcessor::scale_to_original(bbox, size, size);

    // When original and network size are the same, bbox should be unchanged
    EXPECT_EQ(scaled.x, bbox.x);
    EXPECT_EQ(scaled.y, bbox.y);
    EXPECT_EQ(scaled.width, bbox.width);
    EXPECT_EQ(scaled.height, bbox.height);
}

TEST_F(BboxProcessorTest, InvalidBboxSizeThrows) {
    cv::Size image_size(640, 480);
    std::vector<float> bbox = {100.0f, 100.0f}; // Only 2 elements instead of 4
    int network_width = 640;
    int network_height = 640;

    // Should throw due to invalid bbox size
    EXPECT_THROW(
        {
            auto unused = BBoxProcessor::calculate_bounding_box(image_size, bbox, network_width, network_height);
            (void)unused;
        },
        std::invalid_argument);
}
