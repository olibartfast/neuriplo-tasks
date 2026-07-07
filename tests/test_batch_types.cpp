#include "neuriplo/tasks/core/batch_types.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

TEST(BatchTypesTest, BatchRequestHoldsImages) {
    BatchRequest request;
    request.images = {neuriplo_tasks::vision_test::makeImage(10, 10, 3, 0),
                      neuriplo_tasks::vision_test::makeImage(30, 20, 3, 0)};

    EXPECT_EQ(request.images.size(), 2u);
    EXPECT_EQ(request.images[0].rows(), 10);
    EXPECT_EQ(request.images[1].cols(), 30);
}

TEST(BatchTypesTest, BatchPreprocessOutputDefaultConstruction) {
    BatchPreprocessOutput output;
    EXPECT_TRUE(output.buffers.empty());
    EXPECT_EQ(output.batch_size, 0);
}

TEST(BatchTypesTest, BatchPostprocessOutputDefaultConstruction) {
    BatchPostprocessOutput output;
    EXPECT_TRUE(output.results.empty());
    EXPECT_EQ(output.batch_size, 0);
}

TEST(BatchTypesTest, ImageBatchSizeMatches) {
    BatchRequest request;
    request.images = {neuriplo_tasks::vision_test::makeImage(8, 8, 3, 0)};

    EXPECT_TRUE(imageBatchSizeMatches(request, 1));
    EXPECT_FALSE(imageBatchSizeMatches(request, 0));
    EXPECT_FALSE(imageBatchSizeMatches(request, 2));
    EXPECT_FALSE(imageBatchSizeMatches(request, -1));
}

TEST(BatchTypesTest, PostprocessResultsMatchBatchSize) {
    BatchPostprocessOutput output;
    output.batch_size = 2;
    output.results = {Classification{1.0f, 0.5f}, Classification{2.0f, 0.6f}};

    EXPECT_TRUE(postprocessResultsMatchBatchSize(output));

    output.results.pop_back();
    EXPECT_FALSE(postprocessResultsMatchBatchSize(output));
}

TEST(BatchTypesTest, BatchPreprocessOutputStoresBuffersAndBatchSize) {
    BatchPreprocessOutput output;
    output.buffers = {{1, 2, 3}, {4, 5}};
    output.batch_size = 2;

    EXPECT_EQ(output.buffers.size(), 2u);
    EXPECT_EQ(output.batch_size, 2);
    EXPECT_EQ(output.buffers[0].size(), 3u);
}
