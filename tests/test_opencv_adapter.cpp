#include "../src/core/vision/image_ops.hpp"
#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {

using neuriplo_tasks::vision::Image;
using neuriplo_tasks::vision::PixelType;
using neuriplo_tasks::vision::opencv::copyFromCvMat;
using neuriplo_tasks::vision::opencv::toCvMat;
using neuriplo_tasks::vision::opencv::toImageView;

TEST(OpenCvAdapterTest, WrapsContinuousMatWithoutCopying) {
    cv::Mat mat(2, 3, CV_8UC3, cv::Scalar(1, 2, 3));

    const auto view = toImageView(mat);

    EXPECT_EQ(view.width(), 3);
    EXPECT_EQ(view.height(), 2);
    EXPECT_EQ(view.channels(), 3);
    EXPECT_EQ(view.pixelType(), PixelType::UInt8);
    EXPECT_EQ(view.raw(), mat.data);
}

TEST(OpenCvAdapterTest, CopiesNonContiguousMatRows) {
    cv::Mat backing(3, 5, CV_32FC1);
    for (int row = 0; row < backing.rows; ++row) {
        for (int col = 0; col < backing.cols; ++col) {
            backing.at<float>(row, col) = static_cast<float>(row * 10 + col);
        }
    }
    const cv::Mat roi = backing(cv::Rect(1, 0, 3, 3));
    ASSERT_FALSE(roi.isContinuous());

    const Image image = copyFromCvMat(roi);

    EXPECT_EQ(image.width(), 3);
    EXPECT_EQ(image.height(), 3);
    EXPECT_FLOAT_EQ(image.ptr<float>(2)[1], 22.0F);
    EXPECT_THROW(static_cast<void>(toImageView(roi)), std::invalid_argument);
}

TEST(OpenCvAdapterTest, RoundTripsImageData) {
    Image image(2, 2, 2, PixelType::Int32);
    auto* values = image.data<int32_t>();
    for (int32_t index = 0; index < 8; ++index) {
        values[index] = index * 7;
    }

    const cv::Mat mat = toCvMat(image.view());
    const Image copy = copyFromCvMat(mat);

    EXPECT_EQ(mat.type(), CV_32SC2);
    EXPECT_EQ(copy.sizeBytes(), image.sizeBytes());
    EXPECT_EQ(copy.data<int32_t>()[7], 49);
}

class ResizeParityTest : public testing::TestWithParam<std::tuple<neuriplo_tasks::vision::ops::Interpolation, int>> {};

TEST_P(ResizeParityTest, MatchesOpenCvWithinTolerance) {
    Image source(8, 8, 1, PixelType::Float32);
    for (int row = 0; row < source.height(); ++row) {
        for (int col = 0; col < source.width(); ++col) {
            source.ptr<float>(row)[col] = static_cast<float>((row * 17 + col * 11) % 53);
        }
    }

    const auto [interpolation, cv_interpolation] = GetParam();
    const Image actual = neuriplo_tasks::vision::ops::resize(source, 4, 4, interpolation);
    cv::Mat expected;
    cv::resize(toCvMat(source.view()), expected, cv::Size(4, 4), 0.0, 0.0, cv_interpolation);

    double max_error = 0.0;
    for (int row = 0; row < actual.height(); ++row) {
        for (int col = 0; col < actual.width(); ++col) {
            max_error = std::max(
                max_error, std::abs(static_cast<double>(actual.ptr<float>(row)[col] - expected.at<float>(row, col))));
        }
    }
    const double tolerance = 1.0e-5;
    EXPECT_LE(max_error, tolerance);
}

INSTANTIATE_TEST_SUITE_P(
    Interpolations, ResizeParityTest,
    testing::Values(std::make_tuple(neuriplo_tasks::vision::ops::Interpolation::Linear, cv::INTER_LINEAR),
                    std::make_tuple(neuriplo_tasks::vision::ops::Interpolation::Area, cv::INTER_AREA),
                    std::make_tuple(neuriplo_tasks::vision::ops::Interpolation::Cubic, cv::INTER_CUBIC)));

} // namespace
