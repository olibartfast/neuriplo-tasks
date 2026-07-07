#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "vision_test_utils.hpp"

#include <gtest/gtest.h>

using namespace neuriplo_tasks;

class ConcreteTasksTest : public ::testing::Test {
  protected:
    ModelInfo createValidModelInfo(const std::string& format = "FORMAT_NCHW") {
        ModelInfo info;
        info.input_shapes = {{1, 3, 640, 640}};
        info.input_formats = {format};
        info.input_names = {"images"};
        info.output_names = {"output0"};
        info.input_types = {neuriplo_tasks::PixelType::Float32};
        return info;
    }
};

TEST_F(ConcreteTasksTest, CreateYoloTask) {
    auto info = createValidModelInfo();

    // Test various YOLO strings
    std::vector<std::string> types = {"yolo", "yolov5", "yolov8", "yolov10", "yolo26", "yolo11"};

    for (const auto& type : types) {
        auto task = TaskFactory::createTaskInstance(type, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for type: " << type;
        EXPECT_EQ(task->getTaskType(), TaskType::Detection);
    }
}

TEST_F(ConcreteTasksTest, CreateRtDetrTask) {
    auto info = createValidModelInfo();

    auto task = TaskFactory::createTaskInstance("rtdetr", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);

    task = TaskFactory::createTaskInstance("rtdetr-ultralytics", info);
    ASSERT_NE(task, nullptr);
}

TEST_F(ConcreteTasksTest, CreateClassificationTask) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};

    std::vector<std::string> types = {"resnet", "resnet50", "vit-classifier", "tensorflow-classifier"};

    for (const auto& type : types) {
        auto task = TaskFactory::createTaskInstance(type, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for type: " << type;
        EXPECT_EQ(task->getTaskType(), TaskType::Classification);
    }
}

TEST_F(ConcreteTasksTest, CreateDepthAnythingV2Task) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 518, 518}};
    info.input_formats = {"FORMAT_NCHW"};

    auto task = TaskFactory::createTaskInstance("depth-anything-v2", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::DepthEstimation);
}

TEST_F(ConcreteTasksTest, CreateOwlv2Task) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 960, 960}, {2, 16}, {2, 16}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"pixel_values", "input_ids", "attention_mask"};
    info.output_names = {"pred_boxes", "logits"};
    info.input_types = {neuriplo_tasks::PixelType::Float32, neuriplo_tasks::PixelType::Int32,
                        neuriplo_tasks::PixelType::Int32};

    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("owlv2", info, cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::OpenVocabDetection);
}

TEST_F(ConcreteTasksTest, CreateGroundingDinoTask) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 800, 800}, {1, 256}, {1, 256}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"pixel_values", "input_ids", "attention_mask"};
    info.output_names = {"pred_boxes", "pred_logits"};
    info.input_types = {neuriplo_tasks::PixelType::Float32, neuriplo_tasks::PixelType::Int32,
                        neuriplo_tasks::PixelType::Int32};

    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("groundingdino", info, cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::OpenVocabDetection);
}

TEST_F(ConcreteTasksTest, PreprocessExecution) {
    // Verify that we can call preprocess without crashing
    auto info = createValidModelInfo();
    auto task = TaskFactory::createTaskInstance("yolov8", info);

    neuriplo_tasks::Image img = neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0);
    std::vector<neuriplo_tasks::Image> images = {img};

    auto result = task->preprocess(images);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 1);
}
