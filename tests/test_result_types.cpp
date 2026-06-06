#include "vision-core/core/opencv_interop.hpp"
#include "vision-core/core/result_types.hpp"
#include "vision-core/core/tensor_utils.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <type_traits>

using namespace vision_core;

namespace {

std::string resultTypeName(const Result& result) {
    return visitResult(result, [](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Classification>) {
            return "classification";
        } else if constexpr (std::is_same_v<T, Detection>) {
            return "detection";
        } else if constexpr (std::is_same_v<T, OpenVocabDetection>) {
            return "open_vocab_detection";
        } else if constexpr (std::is_same_v<T, InstanceSegmentation>) {
            return "instance_segmentation";
        } else if constexpr (std::is_same_v<T, OpticalFlow>) {
            return "optical_flow";
        } else if constexpr (std::is_same_v<T, VideoClassification>) {
            return "video_classification";
        } else if constexpr (std::is_same_v<T, PoseEstimation>) {
            return "pose_estimation";
        } else if constexpr (std::is_same_v<T, DepthEstimation>) {
            return "depth_estimation";
        } else if constexpr (std::is_same_v<T, GaussianSplatting>) {
            return "gaussian_splatting";
        } else if constexpr (std::is_same_v<T, ImageUnderstanding>) {
            return "image_understanding";
        }
    });
}

} // namespace

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

TEST_F(ResultTypesTest, TensorElementToFloatCoversSupportedScalarTypes) {
    EXPECT_FLOAT_EQ(tensorElementToFloat(TensorElement{1.25f}), 1.25f);
    EXPECT_FLOAT_EQ(tensorElementToFloat(TensorElement{int32_t{-2}}), -2.0f);
    EXPECT_FLOAT_EQ(tensorElementToFloat(TensorElement{int64_t{3}}), 3.0f);
    EXPECT_FLOAT_EQ(tensorElementToFloat(TensorElement{uint8_t{4}}), 4.0f);
}

TEST_F(ResultTypesTest, TensorElementToIntCoversSupportedScalarTypes) {
    EXPECT_EQ(tensorElementToInt(TensorElement{1.75f}), 1);
    EXPECT_EQ(tensorElementToInt(TensorElement{int32_t{-2}}), -2);
    EXPECT_EQ(tensorElementToInt(TensorElement{int64_t{3}}), 3);
    EXPECT_EQ(tensorElementToInt(TensorElement{uint8_t{4}}), 4);
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
    BoundingBox bbox(100, 200, 50, 75);
    Detection det(bbox, 0.85f, 3);

    EXPECT_FLOAT_EQ(det.class_id, 3.0f);
    EXPECT_FLOAT_EQ(det.class_confidence, 0.85f);
    EXPECT_EQ(det.bbox.x, 100);
    EXPECT_EQ(det.bbox.y, 200);
    EXPECT_EQ(det.bbox.width, 50);
    EXPECT_EQ(det.bbox.height, 75);
}

TEST_F(ResultTypesTest, DetectionInheritsFromClassification) {
    Detection det(BoundingBox(10, 20, 30, 40), 0.9f, 7);

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
    BoundingBox bbox(50, 60, 100, 120);
    InstanceSegmentation seg(bbox, 0.92f, 1);

    EXPECT_FLOAT_EQ(seg.class_id, 1.0f);
    EXPECT_FLOAT_EQ(seg.class_confidence, 0.92f);
    EXPECT_EQ(seg.bbox.x, 50);
    EXPECT_EQ(seg.bbox.y, 60);
}

TEST_F(ResultTypesTest, InstanceSegmentationInheritsFromDetection) {
    InstanceSegmentation seg(BoundingBox(1, 2, 3, 4), 0.8f, 2);

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
    flow.flow = fromCvMat(cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 255)));
    flow.raw_flow = fromCvMat(cv::Mat(100, 100, CV_32FC2));
    flow.max_displacement = 15.5f;

    EXPECT_EQ(flow.flow.rows(), 100);
    EXPECT_EQ(flow.flow.cols(), 100);
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
    Detection det(BoundingBox(1, 2, 3, 4), 0.9f, 5);
    Result result = det;

    EXPECT_TRUE(std::holds_alternative<Detection>(result));
    EXPECT_FALSE(std::holds_alternative<Classification>(result));
}

TEST_F(ResultTypesTest, OpenVocabDetectionDefaultConstruction) {
    OpenVocabDetection det;
    EXPECT_EQ(det.bbox.width, 0);
    EXPECT_EQ(det.bbox.height, 0);
    EXPECT_FLOAT_EQ(det.score, 0.0f);
    EXPECT_EQ(det.prompt_index, -1);
    EXPECT_TRUE(det.label.empty());
}

TEST_F(ResultTypesTest, OpenVocabDetectionParameterizedConstruction) {
    OpenVocabDetection det(BoundingBox(11, 12, 13, 14), 0.77f, 2, "cat");
    EXPECT_EQ(det.bbox.x, 11);
    EXPECT_EQ(det.bbox.y, 12);
    EXPECT_FLOAT_EQ(det.score, 0.77f);
    EXPECT_EQ(det.prompt_index, 2);
    EXPECT_EQ(det.label, "cat");
}

TEST_F(ResultTypesTest, ResultVariantHoldsOpenVocabDetection) {
    OpenVocabDetection det(BoundingBox(0, 1, 2, 3), 0.91f, 1, "vehicle");
    Result result = det;

    EXPECT_TRUE(std::holds_alternative<OpenVocabDetection>(result));
    auto& stored = std::get<OpenVocabDetection>(result);
    EXPECT_EQ(stored.label, "vehicle");
}

TEST_F(ResultTypesTest, ResultVariantHoldsInstanceSegmentation) {
    InstanceSegmentation seg(BoundingBox(10, 20, 30, 40), 0.95f, 2);
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
    depth.depth = fromCvMat(cv::Mat::ones(10, 10, CV_32FC1));
    Result result = depth;

    EXPECT_TRUE(std::holds_alternative<DepthEstimation>(result));
}

TEST_F(ResultTypesTest, VisitResultAcceptsEveryResultAlternative) {
    const std::vector<Result> results = {
        Classification{},      Detection{},      OpenVocabDetection{}, InstanceSegmentation{}, OpticalFlow{},
        VideoClassification{}, PoseEstimation{}, DepthEstimation{},    GaussianSplatting{},    ImageUnderstanding{},
    };
    const std::vector<std::string> expected = {
        "classification",        "detection",        "open_vocab_detection",
        "instance_segmentation", "optical_flow",     "video_classification",
        "pose_estimation",       "depth_estimation", "gaussian_splatting",
        "image_understanding",
    };

    ASSERT_EQ(results.size(), expected.size());
    for (size_t index = 0; index < results.size(); ++index) {
        EXPECT_EQ(resultTypeName(results[index]), expected[index]);
    }
}

TEST_F(ResultTypesTest, VisitResultAllowsMutableVisitors) {
    Result result = Classification{};

    visitResult(result, [](auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Classification>) {
            value.class_id = 42.0f;
        }
    });

    EXPECT_FLOAT_EQ(std::get<Classification>(result).class_id, 42.0f);
}

TEST_F(ResultTypesTest, TaskTypeEnumValues) {
    EXPECT_EQ(static_cast<int>(TaskType::OpticalFlow), 0);
    EXPECT_EQ(static_cast<int>(TaskType::Classification), 1);
    EXPECT_EQ(static_cast<int>(TaskType::Detection), 2);
    EXPECT_EQ(static_cast<int>(TaskType::InstanceSegmentation), 3);
    EXPECT_EQ(static_cast<int>(TaskType::VideoClassification), 4);
    EXPECT_EQ(static_cast<int>(TaskType::PoseEstimation), 5);
    EXPECT_EQ(static_cast<int>(TaskType::DepthEstimation), 6);
    EXPECT_EQ(static_cast<int>(TaskType::OpenVocabDetection), 7);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
