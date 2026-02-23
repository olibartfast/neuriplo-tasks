#include "vision-core/core/model_info.hpp"
#include "vision-core/core/task_factory.hpp"

#include <gtest/gtest.h>

using namespace vision_core;

class ConcreteTasksTest : public ::testing::Test {
  protected:
    ModelInfo createValidModelInfo(const std::string& format = "FORMAT_NCHW") {
        ModelInfo info;
        info.input_shapes = {{1, 3, 640, 640}};
        info.input_formats = {format};
        info.input_names = {"images"};
        info.output_names = {"output0"};
        info.input_types = {CV_32F};
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

TEST_F(ConcreteTasksTest, PreprocessExecution) {
    // Verify that we can call preprocess without crashing
    auto info = createValidModelInfo();
    auto task = TaskFactory::createTaskInstance("yolov8", info);

    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    std::vector<cv::Mat> images = {img};

    auto result = task->preprocess(images);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 1);
}
