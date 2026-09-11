#include "neuriplo/tasks/classification/classification_preprocessor.hpp"
#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/preprocessor.hpp"
#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/optical_flow/optical_flow_task.hpp"
#include "neuriplo/tasks/optical_flow/raft_postprocessor.hpp"
#include "neuriplo/tasks/video_classification/video_classification_task.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace neuriplo_tasks;

namespace {

ModelInfo imageModel(const std::vector<int64_t>& shape, PixelType type, const std::string& name = "images") {
    ModelInfo info;
    info.addInput(name, shape);
    info.input_types.back() = type;
    info.addOutput("output0", {1, 10});
    return info;
}

// Distinct constant channels in BGR order: B=10, G=20, R=30.
Image bgrImage(int width, int height) {
    Image image = vision_test::makeImage(width, height, 3);
    std::uint8_t* data = image.raw();
    for (std::size_t i = 0; i < image.totalPixels(); ++i) {
        data[i * 3] = 10;
        data[i * 3 + 1] = 20;
        data[i * 3 + 2] = 30;
    }
    return image;
}

} // namespace

TEST(ImageInputShape, RankThreeOrMoreCarriesAnImage) {
    EXPECT_TRUE(isImageInputShape({3, 224, 224}));
    EXPECT_TRUE(isImageInputShape({1, 3, 224, 224}));
    EXPECT_TRUE(isImageInputShape({1, 16, 3, 224, 224}));
    EXPECT_FALSE(isImageInputShape({1, 2}));
    EXPECT_FALSE(isImageInputShape({}));
}

TEST(ApplyImageInputType, UInt8EmitsRawPixelsInPlanarOrder) {
    Preprocessor preprocessor(PreprocessConfig{Size(4, 2), ImageFormat::NCHW, DataType::FLOAT32, true, true, true});
    applyImageInputType(preprocessor, imageModel({1, 3, 2, 4}, PixelType::UInt8));

    const auto bytes = preprocessor.preprocess(bgrImage(4, 2).view());

    ASSERT_EQ(bytes.size(), 3U * 2U * 4U);
    // BGR->RGB puts the red plane first; no scaling, no ImageNet statistics.
    EXPECT_EQ(bytes[0], 30);
    EXPECT_EQ(bytes[8], 20);
    EXPECT_EQ(bytes[16], 10);
}

TEST(ApplyImageInputType, UInt8KeepsInterleavedLayout) {
    Preprocessor preprocessor(PreprocessConfig{Size(4, 2), ImageFormat::NHWC, DataType::FLOAT32, true, false, false});
    applyImageInputType(preprocessor, imageModel({1, 2, 4, 3}, PixelType::UInt8));

    const auto bytes = preprocessor.preprocess(bgrImage(4, 2).view());

    ASSERT_EQ(bytes.size(), 4U * 2U * 3U);
    EXPECT_EQ(bytes[0], 10);
    EXPECT_EQ(bytes[1], 20);
    EXPECT_EQ(bytes[2], 30);
}

TEST(ApplyImageInputType, Float32LeavesOutputByteIdentical) {
    const PreprocessConfig config{Size(4, 2), ImageFormat::NCHW, DataType::FLOAT32, true, true, true};
    const Preprocessor untouched(config);
    Preprocessor applied(config);
    applyImageInputType(applied, imageModel({1, 3, 2, 4}, PixelType::Float32));

    const Image image = bgrImage(4, 2);

    EXPECT_EQ(applied.preprocess(image.view()), untouched.preprocess(image.view()));
    EXPECT_EQ(applied.preprocess(image.view()).size(), 3U * 2U * 4U * sizeof(float));
}

TEST(ApplyImageInputType, TensorflowClassifierKeepsUInt8UnderTheDefaultType) {
    TensorflowPreprocessor preprocessor(Size(4, 2));
    applyImageInputType(preprocessor, imageModel({1, 2, 4, 3}, PixelType::Float32));

    EXPECT_EQ(preprocessor.preprocess(bgrImage(4, 2).view()).size(), 4U * 2U * 3U);
}

TEST(ApplyImageInputType, RejectsOtherImagePixelTypesByName) {
    Preprocessor preprocessor(PreprocessConfig{Size(4, 2)});
    try {
        applyImageInputType(preprocessor, imageModel({1, 3, 2, 4}, PixelType::Int32, "pixel_values"));
        FAIL() << "an Int32 image input must be rejected";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("pixel_values"), std::string::npos) << message;
        EXPECT_NE(message.find("Int32"), std::string::npos) << message;
    }
}

TEST(ApplyImageInputType, IgnoresNonImageInputs) {
    ModelInfo info = imageModel({1, 3, 2, 4}, PixelType::UInt8);
    info.addInput("orig_target_sizes", {1, 2});
    info.input_types.back() = PixelType::Int32;
    Preprocessor preprocessor(PreprocessConfig{Size(4, 2)});

    EXPECT_NO_THROW(applyImageInputType(preprocessor, info));
}

TEST(ApplyImageInputType, RejectsImageInputsThatDisagree) {
    ModelInfo info = imageModel({1, 3, 2, 4}, PixelType::UInt8, "left");
    info.addInput("right", {1, 3, 2, 4});
    Preprocessor preprocessor(PreprocessConfig{Size(4, 2)});

    EXPECT_THROW(applyImageInputType(preprocessor, info), std::invalid_argument);
}

TEST(ApplyImageInputType, LetterboxedYoloDetectionEmitsRawPixels) {
    ObjectDetectionTask task(imageModel({1, 3, 8, 8}, PixelType::UInt8), "yolo", 0.25F, 0.45F);

    const auto tensors = task.preprocess({bgrImage(8, 4)});

    ASSERT_EQ(tensors.size(), 1U);
    ASSERT_EQ(tensors[0].size(), 3U * 8U * 8U);
    // The 8x4 frame is centred in rows 2..5 of the 8x8 canvas; red plane first.
    EXPECT_EQ(tensors[0][0], 0);
    EXPECT_EQ(tensors[0][2 * 8], 30);
}

TEST(ApplyImageInputType, RaftCannotEmitRawPixels) {
    try {
        OpticalFlowTask task(imageModel({1, 3, 8, 8}, PixelType::UInt8), "raft");
        FAIL() << "RAFT normalizes to [-1, 1] and must reject a UInt8 input";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("raw pixels"), std::string::npos) << error.what();
    }
}

TEST(ApplyImageInputType, VideoMaeCannotEmitRawPixels) {
    EXPECT_THROW(VideoClassificationTask(imageModel({1, 16, 3, 8, 8}, PixelType::UInt8), "videomae", 5, true),
                 std::invalid_argument);
}
