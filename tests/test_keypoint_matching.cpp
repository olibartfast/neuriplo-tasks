#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"
#include "vision_test_utils.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace neuriplo_tasks;

namespace {

constexpr int kInputSize = 512;
constexpr int kInputChannels = 3;
constexpr int kBytesPerElement = 4;

} // namespace

class KeypointMatchingTest : public ::testing::Test {
  protected:
    ModelInfo createModelInfo(int width = kInputSize, int height = kInputSize) {
        ModelInfo info;
        // LightGlue-style interleaved pair input: two images on the batch axis.
        info.input_shapes = {{2, 1, height, width}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"images"};
        info.output_names = {"keypoints", "matches", "match_scores"};
        info.input_types = {neuriplo_tasks::PixelType::Float32};
        return info;
    }

    std::pair<neuriplo_tasks::Image, neuriplo_tasks::Image> createTestPair(int width = 200, int height = 150) {
        return {neuriplo_tasks::vision_test::makeImage(width, height, 3, 0),
                neuriplo_tasks::vision_test::makeImage(width, height, 3, 0)};
    }
};

TEST_F(KeypointMatchingTest, FactoryRoutesModelFamilies) {
    const auto info = createModelInfo();
    const std::vector<std::string> aliases = {"superglue",    "superglue-2020", " light glue ",        "LightGlue",
                                              "lightglue_v2", "matchanything",  "MatchAnything-EloFTR"};

    for (const auto& alias : aliases) {
        auto task = TaskFactory::createTaskInstance(alias, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for alias: " << alias;
        EXPECT_EQ(task->getTaskType(), TaskType::KeypointMatching) << alias;
        EXPECT_EQ(task->getRequiredFrames(), 2) << alias;
    }
}

TEST_F(KeypointMatchingTest, UnknownFamilyIsRejected) {
    EXPECT_THROW(TaskFactory::createTaskInstance("sift", createModelInfo()), std::invalid_argument);
    EXPECT_THROW(TaskFactory::createTaskInstance("keypointmatching", createModelInfo()), std::invalid_argument);
}

TEST_F(KeypointMatchingTest, PreprocessRequiresExactlyTwoImages) {
    auto task = TaskFactory::createTaskInstance("superglue", createModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair();

    EXPECT_THROW(task->preprocess({image1}), std::invalid_argument);
    EXPECT_THROW(task->preprocess({image1, image2, image1}), std::invalid_argument);
    EXPECT_THROW(task->preprocess({}), std::invalid_argument);
}

TEST_F(KeypointMatchingTest, PreprocessRejectsEmptyImageInPair) {
    auto task = TaskFactory::createTaskInstance("superglue", createModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair();
    neuriplo_tasks::Image empty_image;

    EXPECT_THROW(task->preprocess({empty_image, image2}), std::invalid_argument);
    EXPECT_THROW(task->preprocess({image1, empty_image}), std::invalid_argument);
}

TEST_F(KeypointMatchingTest, PreprocessReturnsOneBufferPerImage) {
    auto task = TaskFactory::createTaskInstance("lightglue", createModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair();
    const auto buffers = task->preprocess({image1, image2});

    ASSERT_EQ(buffers.size(), 2U);
    const auto expected_size = static_cast<std::size_t>(kInputChannels) * kInputSize * kInputSize * kBytesPerElement;
    EXPECT_EQ(buffers[0].size(), expected_size);
    EXPECT_EQ(buffers[1].size(), expected_size);
}

TEST_F(KeypointMatchingTest, PreprocessorPairDirect) {
    KeypointMatchingPreprocessor preprocessor(Size(kInputSize, kInputSize));
    auto [image1, image2] = createTestPair();

    const auto buffers = preprocessor.preprocess_pair(image1.view(), image2.view());

    ASSERT_EQ(buffers.size(), 2U);
    EXPECT_FALSE(buffers[0].empty());
    EXPECT_EQ(buffers[0].size(), buffers[1].size());
}

TEST_F(KeypointMatchingTest, PreprocessorRejectsEmptyImage) {
    KeypointMatchingPreprocessor preprocessor(Size(kInputSize, kInputSize));
    auto [image1, image2] = createTestPair();
    neuriplo_tasks::Image empty_image;

    EXPECT_THROW({ (void)preprocessor.preprocess_pair(empty_image.view(), image2.view()); }, std::invalid_argument);
    EXPECT_THROW({ (void)preprocessor.preprocess_pair(image1.view(), empty_image.view()); }, std::invalid_argument);
}

TEST_F(KeypointMatchingTest, PostprocessWithEmptyTensorsReturnsEmpty) {
    auto task = TaskFactory::createTaskInstance("superglue", createModelInfo());
    ASSERT_NE(task, nullptr);

    EXPECT_TRUE(task->postprocess(Size(200, 150), {}).empty());
}

TEST_F(KeypointMatchingTest, DecodingIsNotImplementedYet) {
    const std::vector<std::string> aliases = {"superglue", "lightglue", "matchanything"};
    const std::vector<Tensor> tensors = {Tensor({TensorElement{0.0f}}, {1, 1})};

    for (const auto& alias : aliases) {
        auto task = TaskFactory::createTaskInstance(alias, createModelInfo());
        ASSERT_NE(task, nullptr) << alias;
        EXPECT_THROW(task->postprocess(Size(200, 150), tensors), std::logic_error) << alias;
    }
}
