#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/lightglue_postprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/lightglue_preprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/superglue_postprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/superglue_preprocessor.hpp"
#include "vision_test_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
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
    auto task = TaskFactory::createTaskInstance("matchanything", createModelInfo());
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

TEST_F(KeypointMatchingTest, RemainingFamiliesStillThrowInsteadOfGuessing) {
    const std::vector<std::string> aliases = {"matchanything"};
    const std::vector<Tensor> tensors = {Tensor({TensorElement{0.0f}}, {1, 1})};

    for (const auto& alias : aliases) {
        auto task = TaskFactory::createTaskInstance(alias, createModelInfo());
        ASSERT_NE(task, nullptr) << alias;
        EXPECT_THROW(task->postprocess(Size(200, 150), tensors), std::logic_error) << alias;
    }
}

namespace {

constexpr int kSuperGlueWidth = 640;
constexpr int kSuperGlueHeight = 480;

ModelInfo createSuperGlueModelInfo() {
    ModelInfo info;
    // transformers SuperGlueForKeypointMatching export: pair on a single input.
    info.input_shapes = {{1, 2, 3, kSuperGlueHeight, kSuperGlueWidth}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"pixel_values"};
    info.output_names = {"keypoints", "matches", "matching_scores", "mask"};
    info.input_types = {neuriplo_tasks::PixelType::Float32};
    return info;
}

Tensor makeFloatTensor(const std::vector<float>& values, const std::vector<int64_t>& shape) {
    std::vector<TensorElement> elements;
    elements.reserve(values.size());
    for (float value : values) {
        elements.emplace_back(value);
    }
    return Tensor(std::move(elements), shape);
}

Tensor makeIntTensor(const std::vector<int64_t>& values, const std::vector<int64_t>& shape) {
    std::vector<TensorElement> elements;
    elements.reserve(values.size());
    for (int64_t value : values) {
        elements.emplace_back(value);
    }
    return Tensor(std::move(elements), shape);
}

/**
 * @brief Four keypoints per image, lifted from the verified
 * magic-leap-community/superglue_outdoor export (reference_0.npz)
 *
 * Matches mirror the export: index 2 of the target is a mutual match for
 * source 0, source 1 points at masked-out target padding, source 2 is
 * unmatched, and source 3 points back at target 0.
 */
std::vector<Tensor> makeSuperGlueOutputs() {
    const std::vector<float> keypoints = {// source: x / width, y / height
                                          0.734375F, 0.04791666567325592F, 0.753125011920929F, 0.0520833320915699F,
                                          0.801562488079071F, 0.0729166641831398F, 0.75F, 0.01875000074505806F,
                                          // target
                                          0.734375F, 0.05000000074505806F, 0.753125011920929F, 0.0520833320915699F,
                                          0.7984374761581421F, 0.07083333283662796F, 0.7421875F, 0.04583333432674408F};
    const std::vector<int64_t> matches = {2, 3, -1, 0, 0, 1, 2, -1};
    const std::vector<float> scores = {0.988884687423706F, 0.9874908924102783F, 0.0F, 0.4000000059604645F,
                                       0.988884687423706F, 0.9874908924102783F, 0.0F, 0.0F};
    const std::vector<int64_t> mask = {1, 1, 0, 0, 1, 1, 1, 0};

    return {makeFloatTensor(keypoints, {1, 2, 4, 2}), makeIntTensor(matches, {1, 2, 4}),
            makeFloatTensor(scores, {1, 2, 4}), makeIntTensor(mask, {1, 2, 4})};
}

} // namespace

TEST_F(KeypointMatchingTest, SuperGluePreprocessPacksOnePairBuffer) {
    auto task = TaskFactory::createTaskInstance("superglue", createSuperGlueModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair();
    const auto buffers = task->preprocess({image1, image2});

    // One model input, so one buffer: [2, 3, H, W] float32.
    ASSERT_EQ(buffers.size(), 1U);
    const auto expected_size = static_cast<std::size_t>(2 * 3 * kSuperGlueHeight * kSuperGlueWidth) * kBytesPerElement;
    EXPECT_EQ(buffers[0].size(), expected_size);
}

TEST_F(KeypointMatchingTest, SuperGluePreprocessAppliesGrayscaleAndUnitScaling) {
    SuperGluePreprocessor preprocessor(Size(2, 2));
    neuriplo_tasks::Image source = neuriplo_tasks::vision_test::makeImage(2, 2, 3, 0);
    // BGR input: red, blue, green, white.
    const std::uint8_t pixels[4][3] = {{0, 0, 255}, {255, 0, 0}, {0, 255, 0}, {255, 255, 255}};
    for (std::size_t p = 0; p < 4; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            source.raw()[p * 3 + c] = pixels[p][c];
        }
    }
    neuriplo_tasks::Image target = neuriplo_tasks::vision_test::makeImage(2, 2, 3, 0);

    const auto buffers = preprocessor.preprocess_pair(source.view(), target.view());
    ASSERT_EQ(buffers.size(), 1U);
    ASSERT_EQ(buffers[0].size(), static_cast<std::size_t>(2 * 3 * 2 * 2) * sizeof(float));

    const auto* values = reinterpret_cast<const float*>(buffers[0].data());
    constexpr std::size_t plane = 4;
    const float expected[plane] = {0.2989F, 0.1140F, 0.5870F, 0.9999F};
    for (std::size_t p = 0; p < plane; ++p) {
        EXPECT_NEAR(values[p], expected[p], 1e-6F) << "pixel " << p;
        // Channels are replicated planes of the same grayscale value.
        EXPECT_FLOAT_EQ(values[plane + p], values[p]);
        EXPECT_FLOAT_EQ(values[2 * plane + p], values[p]);
    }
    for (std::size_t p = 0; p < 3 * plane; ++p) {
        EXPECT_FLOAT_EQ(values[3 * plane + p], 0.0F) << "target slot " << p;
    }
}

TEST_F(KeypointMatchingTest, SuperGlueDecodesVerifiedExportValues) {
    SuperGluePostprocessor postprocessor(0.0F);
    const KeypointMatching result = postprocessor.postprocess(makeSuperGlueOutputs(), Size(1024, 768), Size(800, 600));

    ASSERT_EQ(result.source_keypoints.size(), 2U); // mask drops source 2 and 3
    ASSERT_EQ(result.target_keypoints.size(), 3U); // mask drops target 3

    // Keypoints are normalized by width/height, then scaled to the original image.
    EXPECT_NEAR(result.source_keypoints[0].x, 752.0F, 1e-3F);
    EXPECT_NEAR(result.source_keypoints[0].y, 36.8F, 1e-3F);
    EXPECT_NEAR(result.source_keypoints[1].x, 771.2F, 1e-3F);
    EXPECT_NEAR(result.source_keypoints[1].y, 40.0F, 1e-3F);
    EXPECT_NEAR(result.target_keypoints[2].x, 638.75F, 1e-3F);
    EXPECT_NEAR(result.target_keypoints[2].y, 42.5F, 1e-3F);

    // source 0 -> target 2 only: source 1 points at masked-out target padding.
    ASSERT_EQ(result.matches.size(), 1U);
    EXPECT_EQ(result.matches[0].source_index, 0);
    EXPECT_EQ(result.matches[0].target_index, 2);
    EXPECT_NEAR(result.matches[0].score, 0.988884687423706F, 1e-6F);
}

TEST_F(KeypointMatchingTest, SuperGlueDropsMatchesBelowThreshold) {
    const auto tensors = makeSuperGlueOutputs();

    SuperGluePostprocessor generous(0.0F);
    EXPECT_EQ(generous.postprocess(tensors, Size(1024, 768), Size(800, 600)).matches.size(), 1U);

    SuperGluePostprocessor strict(0.995F);
    EXPECT_TRUE(strict.postprocess(tensors, Size(1024, 768), Size(800, 600)).matches.empty());
}

TEST_F(KeypointMatchingTest, SuperGlueIgnoresMaskWhenAbsent) {
    auto tensors = makeSuperGlueOutputs();
    tensors.pop_back(); // no mask output

    SuperGluePostprocessor postprocessor(0.0F);
    const KeypointMatching result = postprocessor.postprocess(tensors, Size(1024, 768), Size(800, 600));

    EXPECT_EQ(result.source_keypoints.size(), 4U);
    EXPECT_EQ(result.target_keypoints.size(), 4U);
    // source 0 -> 2 and source 3 -> 0 both survive; source 1 -> 3 now targets a real keypoint.
    EXPECT_EQ(result.matches.size(), 3U);
}

TEST_F(KeypointMatchingTest, SuperGlueDecodesRegardlessOfKeypointAndScoreOrder) {
    // keypoints and matching_scores are identified by shape and element type,
    // so their position in the tensor list does not matter. (matches and mask
    // share a shape and stay positional: matches first.)
    auto tensors = makeSuperGlueOutputs();
    std::swap(tensors[0], tensors[2]);

    SuperGluePostprocessor postprocessor(0.0F);
    const KeypointMatching result = postprocessor.postprocess(tensors, Size(1024, 768), Size(800, 600));

    EXPECT_EQ(result.source_keypoints.size(), 2U);
    ASSERT_EQ(result.matches.size(), 1U);
    EXPECT_EQ(result.matches[0].target_index, 2);
}

TEST_F(KeypointMatchingTest, SuperGlueRejectsMissingOrInconsistentOutputs) {
    SuperGluePostprocessor postprocessor(0.0F);
    auto tensors = makeSuperGlueOutputs();

    // No keypoints output at all.
    std::vector<Tensor> no_keypoints = {tensors[1], tensors[2], tensors[3]};
    EXPECT_THROW(postprocessor.postprocess(no_keypoints, Size(1024, 768), Size(800, 600)), std::invalid_argument);

    // Keypoints shape says 4 keypoints but the payload is short.
    std::vector<Tensor> short_keypoints = tensors;
    short_keypoints[0] = makeFloatTensor(std::vector<float>(6, 0.0F), {1, 2, 4, 2});
    EXPECT_THROW(postprocessor.postprocess(short_keypoints, Size(1024, 768), Size(800, 600)), std::invalid_argument);

    // Matches report a different N than keypoints.
    std::vector<Tensor> mismatched = tensors;
    mismatched[1] = makeIntTensor(std::vector<int64_t>(6, -1), {1, 2, 3});
    EXPECT_THROW(postprocessor.postprocess(mismatched, Size(1024, 768), Size(800, 600)), std::invalid_argument);
}

TEST_F(KeypointMatchingTest, SuperGlueTaskScalesToRecordedPairSizes) {
    auto task = TaskFactory::createTaskInstance("superglue", createSuperGlueModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair(200, 150);
    (void)task->preprocess({image1, image2});

    const auto results = task->postprocess(Size(200, 150), makeSuperGlueOutputs());
    ASSERT_EQ(results.size(), 1U);

    const auto& matching = std::get<KeypointMatching>(results[0]);
    ASSERT_EQ(matching.source_keypoints.size(), 2U);
    EXPECT_NEAR(matching.source_keypoints[0].x, 0.734375F * 200.0F, 1e-3F);
    EXPECT_NEAR(matching.source_keypoints[0].y, 0.04791666567325592F * 150.0F, 1e-3F);
    // TaskConfig's default confidence_threshold (0.25) keeps the 0.99 match.
    ASSERT_EQ(matching.matches.size(), 1U);
}

namespace {

// Three keypoints per image, coordinates lifted from the verified
// thomasonzhou/superpoint-lightglue export (512x512 input space).
std::vector<Tensor> makeLightGlueOutputs() {
    const std::vector<int64_t> keypoints = {325, 208, 313, 315, 401, 120, 347, 320, 332, 429, 210, 90};
    const std::vector<int64_t> matches = {0, 0, 2, 0, 1, 0, 1, 2, 1};
    const std::vector<float> scores = {0.9F, 0.27867886F, 0.9F};
    return {makeIntTensor(keypoints, {2, 3, 2}), makeIntTensor(matches, {3, 3}), makeFloatTensor(scores, {3})};
}

} // namespace

TEST_F(KeypointMatchingTest, LightGluePreprocessPacksOneGrayscalePairBuffer) {
    auto task = TaskFactory::createTaskInstance("lightglue", createModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair();
    const auto buffers = task->preprocess({image1, image2});

    // One model input [2, 1, H, W]: one buffer, grayscale, both images.
    ASSERT_EQ(buffers.size(), 1U);
    EXPECT_EQ(buffers[0].size(), static_cast<std::size_t>(2 * 1 * kInputSize * kInputSize) * kBytesPerElement);
}

TEST_F(KeypointMatchingTest, LightGluePreprocessAppliesGrayscaleAndUnitScaling) {
    LightGluePreprocessor preprocessor(Size(2, 2), 1);
    neuriplo_tasks::Image source = neuriplo_tasks::vision_test::makeImage(2, 2, 3, 0);
    const std::uint8_t pixels[4][3] = {{0, 0, 255}, {255, 0, 0}, {0, 255, 0}, {255, 255, 255}};
    for (std::size_t p = 0; p < 4; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            source.raw()[p * 3 + c] = pixels[p][c];
        }
    }
    neuriplo_tasks::Image target = neuriplo_tasks::vision_test::makeImage(2, 2, 3, 0);

    const auto buffers = preprocessor.preprocess_pair(source.view(), target.view());
    ASSERT_EQ(buffers.size(), 1U);
    ASSERT_EQ(buffers[0].size(), static_cast<std::size_t>(2 * 1 * 2 * 2) * sizeof(float));

    const auto* values = reinterpret_cast<const float*>(buffers[0].data());
    const float expected[4] = {0.2989F, 0.1140F, 0.5870F, 0.9999F};
    for (std::size_t p = 0; p < 4; ++p) {
        EXPECT_NEAR(values[p], expected[p], 1e-6F) << "pixel " << p;
        EXPECT_FLOAT_EQ(values[4 + p], 0.0F) << "target slot " << p;
    }
}

TEST_F(KeypointMatchingTest, LightGlueDecodesVerifiedExportValues) {
    LightGluePostprocessor postprocessor(0.0F, Size(kInputSize, kInputSize));
    const KeypointMatching result = postprocessor.postprocess(makeLightGlueOutputs(), Size(1024, 768), Size(800, 600));

    ASSERT_EQ(result.source_keypoints.size(), 3U);
    ASSERT_EQ(result.target_keypoints.size(), 3U);

    // Keypoints are input-space pixels, rescaled by original/input per axis.
    EXPECT_NEAR(result.source_keypoints[1].x, 313.0F * 1024.0F / 512.0F, 1e-3F);
    EXPECT_NEAR(result.source_keypoints[1].y, 315.0F * 768.0F / 512.0F, 1e-3F);
    EXPECT_NEAR(result.target_keypoints[1].x, 332.0F * 800.0F / 512.0F, 1e-3F);
    EXPECT_NEAR(result.target_keypoints[1].y, 429.0F * 600.0F / 512.0F, 1e-3F);

    // The row with pair index 1 belongs to another pair and is ignored.
    ASSERT_EQ(result.matches.size(), 2U);
    EXPECT_EQ(result.matches[0].source_index, 0);
    EXPECT_EQ(result.matches[0].target_index, 2);
    EXPECT_NEAR(result.matches[0].score, 0.9F, 1e-6F);
    EXPECT_EQ(result.matches[1].source_index, 1);
    EXPECT_EQ(result.matches[1].target_index, 0);
}

TEST_F(KeypointMatchingTest, LightGlueDropsMatchesBelowThreshold) {
    const auto tensors = makeLightGlueOutputs();

    LightGluePostprocessor generous(0.0F, Size(kInputSize, kInputSize));
    EXPECT_EQ(generous.postprocess(tensors, Size(1024, 768), Size(800, 600)).matches.size(), 2U);

    LightGluePostprocessor strict(0.5F, Size(kInputSize, kInputSize));
    const KeypointMatching filtered = strict.postprocess(tensors, Size(1024, 768), Size(800, 600));
    ASSERT_EQ(filtered.matches.size(), 1U);
    EXPECT_FLOAT_EQ(filtered.matches[0].score, 0.9F);
}

TEST_F(KeypointMatchingTest, LightGlueRejectsMalformedTensors) {
    LightGluePostprocessor postprocessor(0.0F, Size(kInputSize, kInputSize));
    const auto tensors = makeLightGlueOutputs();

    std::vector<Tensor> no_keypoints = {tensors[1], tensors[2]};
    EXPECT_THROW(postprocessor.postprocess(no_keypoints, Size(1024, 768), Size(800, 600)), std::invalid_argument);

    std::vector<Tensor> short_matches = tensors;
    short_matches[1] = makeIntTensor(std::vector<int64_t>(6, 0), {3, 3});
    EXPECT_THROW(postprocessor.postprocess(short_matches, Size(1024, 768), Size(800, 600)), std::invalid_argument);
}

TEST_F(KeypointMatchingTest, LightGlueTaskUsesModelInputSizeForRescaling) {
    auto task = TaskFactory::createTaskInstance("lightglue", createModelInfo());
    ASSERT_NE(task, nullptr);

    auto [image1, image2] = createTestPair(200, 150);
    (void)task->preprocess({image1, image2});

    const auto results = task->postprocess(Size(200, 150), makeLightGlueOutputs());
    ASSERT_EQ(results.size(), 1U);

    const auto& matching = std::get<KeypointMatching>(results[0]);
    ASSERT_EQ(matching.source_keypoints.size(), 3U);
    EXPECT_NEAR(matching.source_keypoints[1].x, 313.0F * 200.0F / 512.0F, 1e-3F);
    EXPECT_NEAR(matching.source_keypoints[1].y, 315.0F * 150.0F / 512.0F, 1e-3F);
    // Default confidence_threshold 0.25 keeps the 0.9 and 0.2787 matches.
    EXPECT_EQ(matching.matches.size(), 2U);
}
