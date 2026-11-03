#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/video_classification/video_classification_preprocessor.hpp"
#include "vision_test_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace neuriplo_tasks;

namespace {

ModelInfo createVideoModelInfo(int frames, int height, int width) {
    ModelInfo info;
    info.input_shapes = {{1, frames, 3, height, width}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"pixel_values_videos"};
    info.output_names = {"logits"};
    info.input_types = {neuriplo_tasks::PixelType::Float32};
    return info;
}

// Single BGR pixel; the preprocessors convert BGR to RGB before normalizing.
Image makeBgrPixel(std::uint8_t blue, std::uint8_t green, std::uint8_t red) {
    Image image = vision_test::makeImage(1, 1, 3, 0);
    std::uint8_t* pixel = image.raw();
    pixel[0] = blue;
    pixel[1] = green;
    pixel[2] = red;
    return image;
}

std::vector<float> toFloats(const std::vector<std::uint8_t>& buffer) {
    const auto* data = reinterpret_cast<const float*>(buffer.data());
    return std::vector<float>(data, data + buffer.size() / sizeof(float));
}

} // namespace

TEST(VideoClassificationTest, RoutesNewFamilies) {
    const ModelInfo info = createVideoModelInfo(16, 256, 256);
    const std::vector<std::string> aliases = {"pevideo",
                                              "pe-video",
                                              "PE Video",
                                              "vjepa2",
                                              "vjepa2-vitl-fpc16-256-ssv2",
                                              "vjepa2.1-vitl",
                                              "videoprism",
                                              "videoprism-base-f16r288"};

    for (const auto& alias : aliases) {
        auto task = TaskFactory::createTaskInstance(alias, info);
        ASSERT_NE(task, nullptr) << "alias: " << alias;
        EXPECT_EQ(task->getTaskType(), TaskType::VideoClassification) << "alias: " << alias;
        EXPECT_EQ(task->getRequiredFrames(), 16) << "alias: " << alias;
    }
}

TEST(VideoClassificationTest, RejectsUnknownFamily) {
    EXPECT_THROW(TaskFactory::createTaskInstance("slowfast", createVideoModelInfo(16, 256, 256)),
                 std::invalid_argument);
}

TEST(VideoClassificationTest, PreprocessesOneBufferPerClipAtFamilySize) {
    struct Case {
        const char* alias;
        int height;
        int width;
    };
    const Case cases[] = {{"pevideo", 336, 336}, {"vjepa2", 256, 256}, {"videoprism", 288, 288}};

    for (const Case& test_case : cases) {
        auto task = TaskFactory::createTaskInstance(test_case.alias,
                                                    createVideoModelInfo(16, test_case.height, test_case.width));
        ASSERT_NE(task, nullptr) << test_case.alias;

        std::vector<Image> frames;
        for (int i = 0; i < 4; ++i) {
            frames.push_back(vision_test::makeImage(64, 48, 3, 10));
        }

        const auto buffers = task->preprocess(frames);
        ASSERT_EQ(buffers.size(), 1U) << test_case.alias;
        const auto expected = static_cast<std::size_t>(16) * 3 * static_cast<std::size_t>(test_case.height) *
                              static_cast<std::size_t>(test_case.width) * sizeof(float);
        EXPECT_EQ(buffers[0].size(), expected) << test_case.alias;
    }
}

TEST(VideoClassificationTest, VideoPrismRescalesWithoutNormalizing) {
    VideoPrismPreprocessor preprocessor(Size(1, 1));
    const Image image = makeBgrPixel(255, 0, 0); // blue

    const std::vector<float> values = toFloats(preprocessor.preprocess(image.view()));
    ASSERT_EQ(values.size(), 3U);
    EXPECT_NEAR(values[0], 0.0F, 1e-5F); // R
    EXPECT_NEAR(values[1], 0.0F, 1e-5F); // G
    EXPECT_NEAR(values[2], 1.0F, 1e-5F); // B, no mean/std subtraction
}

TEST(VideoClassificationTest, PeVideoNormalizesWithHalfMean) {
    PeVideoPreprocessor preprocessor(Size(1, 1));
    const Image image = makeBgrPixel(255, 0, 0); // blue

    const std::vector<float> values = toFloats(preprocessor.preprocess(image.view()));
    ASSERT_EQ(values.size(), 3U);
    EXPECT_NEAR(values[0], -1.0F, 1e-5F);
    EXPECT_NEAR(values[1], -1.0F, 1e-5F);
    EXPECT_NEAR(values[2], 1.0F, 1e-5F);
}

TEST(VideoClassificationTest, Vjepa2NormalizesWithImageNetStatistics) {
    Vjepa2Preprocessor preprocessor(Size(1, 1));
    const Image image = makeBgrPixel(255, 0, 0); // blue

    const std::vector<float> values = toFloats(preprocessor.preprocess(image.view()));
    ASSERT_EQ(values.size(), 3U);
    EXPECT_NEAR(values[0], (0.0F - 0.485F) / 0.229F, 1e-4F);
    EXPECT_NEAR(values[1], (0.0F - 0.456F) / 0.224F, 1e-4F);
    EXPECT_NEAR(values[2], (1.0F - 0.406F) / 0.225F, 1e-4F);
}

TEST(VideoClassificationTest, PostprocessesLogitsToTopK) {
    auto task = TaskFactory::createTaskInstance("vjepa2", createVideoModelInfo(16, 256, 256));
    ASSERT_NE(task, nullptr);

    const Tensor logits({TensorElement{0.1F}, TensorElement{2.0F}, TensorElement{0.5F}, TensorElement{1.5F}}, {1, 4});
    const auto results = task->postprocess(Size(256, 256), {logits});

    // One Result per class in the top-k, ordered by score.
    ASSERT_EQ(results.size(), 4U);
    const auto& top = std::get<VideoClassification>(results[0]);
    EXPECT_EQ(top.class_id, 1.0F);
    EXPECT_NEAR(top.class_confidence, 0.5052F, 1e-3F);
    EXPECT_EQ(std::get<VideoClassification>(results[1]).class_id, 3.0F);
}
