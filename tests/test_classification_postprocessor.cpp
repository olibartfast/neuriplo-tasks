#include "vision-core/classification/classification_postprocessor.hpp"
#include "vision-core/classification/tensorflow_postprocessor.hpp"
#include "vision-core/classification/torchvision_postprocessor.hpp"
#include "vision-core/classification/vit_postprocessor.hpp"

#include <gtest/gtest.h>
#include <variant>
#include <vector>

using namespace vision_core;

class ClassificationPostprocessorTest : public ::testing::Test {
  protected:
    std::vector<TensorElement> createLogits(const std::vector<float>& values) {
        std::vector<TensorElement> elements;
        for (float v : values) {
            elements.push_back(v);
        }
        return elements;
    }
};

TEST_F(ClassificationPostprocessorTest, DefaultPostprocessor) {
    DefaultClassificationPostprocessor processor(3, true); // top_k=3, apply_softmax=true

    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 3.0f, 0.1f};
    std::vector<int64_t> shape = {1, 5};

    auto results = processor.postprocess(createLogits(logits), shape);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].class_id, 3); // 3.0f is max
    EXPECT_EQ(results[1].class_id, 1); // 2.0f is second
    EXPECT_EQ(results[2].class_id, 0); // 1.0f is third

    // Check if softmax was applied (sum roughly 1.0)
    float sum = 0.0f;
    // Note: we can't easily check sum of top-k, but we can check if values are in [0, 1]
    for (const auto& res : results) {
        EXPECT_GE(res.class_confidence, 0.0f);
        EXPECT_LE(res.class_confidence, 1.0f);
    }
}

TEST_F(ClassificationPostprocessorTest, TorchvisionPostprocessor) {
    TorchvisionPostprocessor processor(1, true);

    std::vector<float> logits = {10.0f, 5.0f, 2.0f};
    std::vector<int64_t> shape = {1, 3};

    auto results = processor.postprocess(createLogits(logits), shape);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].class_id, 0);
    EXPECT_GT(results[0].class_confidence, 0.9f); // 10.0 vs 5.0 should be high confidence
}

TEST_F(ClassificationPostprocessorTest, ViTPostprocessor) {
    ViTPostprocessor processor(2, false); // No softmax

    std::vector<float> logits = {0.1f, 0.9f, 0.3f, 0.8f};
    std::vector<int64_t> shape = {1, 4};

    auto results = processor.postprocess(createLogits(logits), shape);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].class_id, 1);
    EXPECT_EQ(results[0].class_confidence, 0.9f);
    EXPECT_EQ(results[1].class_id, 3);
    EXPECT_EQ(results[1].class_confidence, 0.8f);
}

TEST_F(ClassificationPostprocessorTest, TensorflowPostprocessor) {
    TensorflowPostprocessor processor(5, true);

    // Simulate a case where we have more classes than top_k
    std::vector<float> logits(10, 0.0f);
    logits[5] = 5.0f;
    logits[2] = 3.0f;

    std::vector<int64_t> shape = {1, 10};

    auto results = processor.postprocess(createLogits(logits), shape);

    ASSERT_GE(results.size(), 2);
    EXPECT_EQ(results[0].class_id, 5);
    EXPECT_EQ(results[1].class_id, 2);
}

TEST_F(ClassificationPostprocessorTest, EmptyInput) {
    DefaultClassificationPostprocessor processor(1, true);
    auto results = processor.postprocess({}, {});
    EXPECT_TRUE(results.empty());
}

TEST_F(ClassificationPostprocessorTest, BatchedLogitsReturnOnePerBatchIndex) {
    TorchvisionPostprocessor processor(5, false);

    std::vector<float> logits = {0.1f, 0.2f, 3.0f, 0.0f, 0.0f, 1.0f, 4.0f, 0.5f};
    std::vector<int64_t> shape = {2, 4};

    auto results = processor.postprocess(createLogits(logits), shape);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].class_id, 2);
    EXPECT_EQ(results[1].class_id, 2);
}
