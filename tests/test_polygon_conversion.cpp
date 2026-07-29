#include "neuriplo/tasks/core/image_matrix.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"
#include "neuriplo/tasks/instance_segmentation/polygon_conversion.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>

using namespace neuriplo_tasks;

TEST(PolygonConversionTest, ConvertsRectangleAndRemovesCollinearPoints) {
    vision::Image mask = vision::Image::zeros(6, 5, 1, PixelType::UInt8);
    for (int y = 1; y < 4; ++y) {
        for (int x = 2; x < 5; ++x) {
            mask.ptr<uint8_t>(y)[x] = 255;
        }
    }

    const auto polygons = maskToPolygons(fromImage(std::move(mask)));

    ASSERT_EQ(polygons.size(), 1U);
    EXPECT_EQ(polygons[0].exterior.size(), 4U);
    EXPECT_TRUE(polygons[0].holes.empty());
}

TEST(PolygonConversionTest, PreservesDisconnectedRegionsAndHoles) {
    vision::Image mask = vision::Image::zeros(9, 7, 1, PixelType::UInt8);
    for (int y = 1; y < 6; ++y) {
        for (int x = 1; x < 6; ++x) {
            mask.ptr<uint8_t>(y)[x] = 255;
        }
    }
    mask.ptr<uint8_t>(3)[3] = 0;
    mask.ptr<uint8_t>(1)[7] = 255;

    const auto polygons = maskToPolygons(fromImage(std::move(mask)));

    ASSERT_EQ(polygons.size(), 2U);
    const auto with_hole = std::find_if(polygons.begin(), polygons.end(),
                                        [](const SegmentationPolygon& polygon) { return !polygon.holes.empty(); });
    ASSERT_NE(with_hole, polygons.end());
    ASSERT_EQ(with_hole->holes.size(), 1U);
    EXPECT_EQ(with_hole->holes[0].size(), 4U);
}

TEST(PolygonConversionTest, ConvexifiesConcaveExterior) {
    vision::Image mask = vision::Image::zeros(6, 6, 1, PixelType::UInt8);
    for (int y = 1; y < 5; ++y) {
        mask.ptr<uint8_t>(y)[1] = 255;
    }
    for (int x = 1; x < 5; ++x) {
        mask.ptr<uint8_t>(4)[x] = 255;
    }

    const auto polygons = maskToPolygons(fromImage(std::move(mask)));

    ASSERT_EQ(polygons.size(), 1U);
    ASSERT_GE(polygons[0].exterior.size(), 3U);
    const auto& ring = polygons[0].exterior;
    for (size_t index = 0; index < ring.size(); ++index) {
        const auto& origin = ring[index];
        const auto& a = ring[(index + 1) % ring.size()];
        const auto& b = ring[(index + 2) % ring.size()];
        const float turn = (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
        EXPECT_GE(turn, 0.0F);
    }
}

TEST(PolygonConversionTest, RejectsNonUint8Masks) {
    vision::Image mask = vision::Image::zeros(2, 2, 1, PixelType::Float32);
    EXPECT_THROW(maskToPolygons(fromImage(std::move(mask))), std::invalid_argument);
}
