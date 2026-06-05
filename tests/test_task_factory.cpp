#include "vision-core/core/model_info.hpp"
#include "vision-core/core/task_factory.hpp"

#include <future>
#include <gtest/gtest.h>
#include <thread>
#include <utility>
#include <vector>

using namespace vision_core;

// Minimal test task for registration tests
class TestTask : public TaskInterface {
  public:
    TestTask(const ModelInfo& model_info) : TaskInterface(model_info) {}

    TaskType getTaskType() override { return TaskType::Detection; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override { return {}; }

    std::vector<Result> postprocess(const cv::Size&, const std::vector<Tensor>&) override { return {}; }
};

class TaskFactoryTest : public ::testing::Test {
  protected:
    ModelInfo createValidModelInfo() {
        ModelInfo info;
        info.input_shapes = {{1, 3, 640, 640}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"images"};
        info.output_names = {"output0"};
        info.input_types = {CV_32F};
        info.max_batch_size_ = 1;
        info.batch_size_ = 1;
        return info;
    }

    ModelInfo createVideoModelInfo() {
        auto info = createValidModelInfo();
        info.input_shapes = {{1, 16, 3, 224, 224}};
        return info;
    }

    void expectAliasesRouteTo(const std::vector<const char*>& aliases, TaskType task_type) {
        auto info = createValidModelInfo();

        for (const char* alias : aliases) {
            auto task = TaskFactory::createTaskInstance(alias, info);
            ASSERT_NE(task, nullptr) << "null for alias: " << alias;
            EXPECT_EQ(task->getTaskType(), task_type) << "alias: " << alias;
        }
    }
};

TEST_F(TaskFactoryTest, ValidateInputSizesEmpty) {
    ModelInfo info;
    info.input_shapes = {};

    EXPECT_THROW({ TaskFactory::createTaskInstance("yolov8", info); }, InputDimensionError);
}

TEST_F(TaskFactoryTest, ValidateInputSizesNegative) {
    ModelInfo info;
    info.input_shapes = {{1, 3, -640, 640}};
    info.input_formats = {"FORMAT_NCHW"};

    EXPECT_THROW({ TaskFactory::createTaskInstance("yolov8", info); }, InputDimensionError);
}

TEST_F(TaskFactoryTest, UnrecognizedModelType) {
    auto info = createValidModelInfo();

    EXPECT_THROW({ TaskFactory::createTaskInstance("unknown_model", info); }, std::invalid_argument);
}

TEST_F(TaskFactoryTest, EmptyModelType) {
    auto info = createValidModelInfo();

    EXPECT_THROW({ TaskFactory::createTaskInstance("", info); }, std::invalid_argument);
}

TEST_F(TaskFactoryTest, InvalidInputDimensions) {
    ModelInfo info;
    info.input_shapes = {{1, 0}};
    info.input_formats = {"FORMAT_NCHW"};

    EXPECT_THROW({ TaskFactory::createTaskInstance("yolov8", info); }, InputDimensionError);
}

TEST_F(TaskFactoryTest, CreateValidYoloTask) {
    auto info = createValidModelInfo();

    // Should be able to create YOLOv8 task
    auto task = TaskFactory::createTaskInstance("yolov8", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, CreateValidRtDetrTask) {
    auto info = createValidModelInfo();

    // Should be able to create RT-DETR task
    auto task = TaskFactory::createTaskInstance("rtdetr", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, CreateValidOwlv2Task) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 960, 960}, {2, 16}, {2, 16}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"pixel_values", "input_ids", "attention_mask"};
    info.output_names = {"pred_boxes", "logits"};
    info.input_types = {CV_32F, CV_32S, CV_32S};

    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("owlv2", info, cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::OpenVocabDetection);
}

TEST_F(TaskFactoryTest, CreateValidClassificationTask) {
    auto info = createValidModelInfo();
    // Update for classification
    info.input_shapes = {{1, 3, 224, 224}};

    // Should be able to create ResNet task
    auto task = TaskFactory::createTaskInstance("resnet50", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Classification);
}

TEST_F(TaskFactoryTest, CreateValidDepthEstimationTask) {
    auto info = createValidModelInfo();
    info.input_shapes = {{1, 3, 518, 518}};

    auto task = TaskFactory::createTaskInstance("depth_anything_v2", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::DepthEstimation);
}

// Regression guard: the unreachable `normalized == "lgm-mini"` branch was
// removed from TaskFactory. normalizeModelType strips hyphens, so the
// hyphenated form must still route through the "lgmmini" branch.
TEST_F(TaskFactoryTest, GaussianSplattingHyphenatedAliasNormalizes) {
    auto info = createValidModelInfo();
    info.input_shapes = {{1, 3, 256, 256}};

    for (const char* alias : {"lgm-mini", "lgmmini", "LGM-Mini"}) {
        auto task = TaskFactory::createTaskInstance(alias, info);
        ASSERT_NE(task, nullptr) << "null for alias: " << alias;
        EXPECT_EQ(task->getTaskType(), TaskType::GaussianSplatting) << "alias: " << alias;
    }
}

TEST_F(TaskFactoryTest, RtDetrUltralyticsLongFormAlias) {
    auto info = createValidModelInfo();

    auto task = TaskFactory::createTaskInstance("rtdetrultralytics", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, FactoryRoutesEveryTaskDomain) {
    const std::vector<std::pair<const char*, TaskType>> aliases = {
        {"yolov8", TaskType::Detection},
        {"yoloseg", TaskType::InstanceSegmentation},
        {"rfdetrseg", TaskType::InstanceSegmentation},
        {"resnet50", TaskType::Classification},
        {"raft", TaskType::OpticalFlow},
        {"yolov8pose", TaskType::PoseEstimation},
        {"depth-anything-v2", TaskType::DepthEstimation},
        {"owlv2", TaskType::OpenVocabDetection},
        {"lgm", TaskType::GaussianSplatting},
        {"gemma4", TaskType::ImageUnderstanding},
    };

    auto info = createValidModelInfo();
    for (const auto& alias : aliases) {
        auto task = TaskFactory::createTaskInstance(alias.first, info);
        ASSERT_NE(task, nullptr) << "null for alias: " << alias.first;
        EXPECT_EQ(task->getTaskType(), alias.second) << "alias: " << alias.first;
    }

    for (const char* alias : {"videomae", "vivit", "timesformer"}) {
        auto video_task = TaskFactory::createTaskInstance(alias, createVideoModelInfo());
        ASSERT_NE(video_task, nullptr) << "null for alias: " << alias;
        EXPECT_EQ(video_task->getTaskType(), TaskType::VideoClassification) << "alias: " << alias;
    }
}

TEST_F(TaskFactoryTest, NormalizedAliasesRouteToSameTaskType) {
    expectAliasesRouteTo({"YOLO-V8", "yolo_v8", " yolo v8 "}, TaskType::Detection);
    expectAliasesRouteTo({"RT-DETR", "rt_detr", " RT DETR "}, TaskType::Detection);
    expectAliasesRouteTo({"ViT-Classifier", "vit_classifier", " VIT CLASSIFIER "}, TaskType::Classification);
    expectAliasesRouteTo({"Depth-Anything-V2", "depth_anything_v2", " DEPTH ANYTHING V2 "}, TaskType::DepthEstimation);
    expectAliasesRouteTo({"LGM-Mini", "lgm_mini", " LGM MINI "}, TaskType::GaussianSplatting);
    expectAliasesRouteTo({"Image-Understanding", "image_understanding", " IMAGE UNDERSTANDING "},
                         TaskType::ImageUnderstanding);
}

TEST_F(TaskFactoryTest, PrefixAndSubstringAliasesRouteToDocumentedTaskTypes) {
    expectAliasesRouteTo({"ecdet", "ecdet-lite", "edgecrafter-det", "edgecrafter"}, TaskType::Detection);
    expectAliasesRouteTo({"ecseg", "ecseg-lite", "edgecrafter-seg"}, TaskType::InstanceSegmentation);
    expectAliasesRouteTo({"resnet101", "custom-tensorflow-classifier"}, TaskType::Classification);
    expectAliasesRouteTo({"ecpose", "ecpose-lite", "edgecrafter-pose"}, TaskType::PoseEstimation);
    expectAliasesRouteTo({"fast-splat-model"}, TaskType::GaussianSplatting);
}

TEST_F(TaskFactoryTest, SpecificYoloAliasesPrecedeGenericDetection) {
    expectAliasesRouteTo({"yolo-seg", "YOLO_SEG", " yolo seg ", "yolov8-seg"}, TaskType::InstanceSegmentation);
    expectAliasesRouteTo({"yolo-pose", "YOLO_POSE", " yolo pose ", "yolov8-pose"}, TaskType::PoseEstimation);
}

TEST_F(TaskFactoryTest, SpecificDetrSegmentationAliasPrecedesDetection) {
    expectAliasesRouteTo({"rfdetrseg", "RF-DETR-SEG", " rf detr seg "}, TaskType::InstanceSegmentation);
    expectAliasesRouteTo({"rfdetr", "RF-DETR", " rf detr "}, TaskType::Detection);
}

TEST_F(TaskFactoryTest, SpecificEdgeCrafterAliasesPrecedeGenericDetection) {
    expectAliasesRouteTo({"ecseg", "EC-SEG", " edgecrafter seg ", "edgecrafter-lite-seg"},
                         TaskType::InstanceSegmentation);
    expectAliasesRouteTo({"ecpose", "EC-POSE", " edgecrafter pose ", "edgecrafter-lite-pose"},
                         TaskType::PoseEstimation);
    expectAliasesRouteTo({"ecdet", "EC-DET", " edgecrafter det ", "edgecrafter"}, TaskType::Detection);
}

TEST_F(TaskFactoryTest, ImageUnderstandingAliases) {
    auto info = createValidModelInfo();

    for (const char* alias : {"gemma4", "gemma", "llama", "llamacpp", "imageunderstanding"}) {
        auto task = TaskFactory::createTaskInstance(alias, info);
        ASSERT_NE(task, nullptr) << "null for alias: " << alias;
        EXPECT_EQ(task->getTaskType(), TaskType::ImageUnderstanding) << "alias: " << alias;
    }
}

// Note: Actual task creation tests would require implementing the concrete task classes
// These tests verify the factory's error handling and validation logic

TEST_F(TaskFactoryTest, ModelInfoCopy) {
    auto info1 = createValidModelInfo();
    ModelInfo info2 = info1;

    EXPECT_EQ(info1.input_shapes.size(), info2.input_shapes.size());
    EXPECT_EQ(info1.input_formats.size(), info2.input_formats.size());
    EXPECT_EQ(info1.input_names.size(), info2.input_names.size());
    EXPECT_EQ(info1.max_batch_size_, info2.max_batch_size_);
}

TEST_F(TaskFactoryTest, ModelInfoDefaultConstruction) {
    ModelInfo info;
    EXPECT_TRUE(info.input_shapes.empty());
    EXPECT_TRUE(info.input_formats.empty());
    EXPECT_TRUE(info.input_names.empty());
    EXPECT_TRUE(info.output_names.empty());
    EXPECT_EQ(info.max_batch_size_, 1);
    EXPECT_EQ(info.batch_size_, 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
