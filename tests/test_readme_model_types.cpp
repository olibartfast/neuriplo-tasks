#include "vision-core/core/model_info.hpp"
#include "vision-core/core/task_config.hpp"
#include "vision-core/core/task_factory.hpp"

#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#ifndef VISION_CORE_README_PATH
#error "VISION_CORE_README_PATH must be defined via target_compile_definitions"
#endif

using namespace vision_core;

namespace {

constexpr const char* kMarkerStart = "<!-- TASKFACTORY_MODEL_LIST:START -->";
constexpr const char* kMarkerEnd = "<!-- TASKFACTORY_MODEL_LIST:END -->";

std::string readReadme() {
    std::ifstream in(VISION_CORE_README_PATH);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string extractBlock(const std::string& readme) {
    const auto start = readme.find(kMarkerStart);
    const auto end = readme.find(kMarkerEnd);
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        return {};
    }
    return readme.substr(start + std::string(kMarkerStart).size(), end - start - std::string(kMarkerStart).size());
}

std::set<std::string> extractQuotedAliases(const std::string& block) {
    static const std::regex kAlias(R"RX("([a-z0-9_-]+)")RX");
    std::set<std::string> out;
    for (std::sregex_iterator it(block.begin(), block.end(), kAlias), e; it != e; ++it) {
        out.insert((*it)[1].str());
    }
    return out;
}

ModelInfo makeValidModelInfo() {
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

} // namespace

TEST(ReadmeModelTypesContract, MarkersArePresent) {
    const std::string readme = readReadme();
    ASSERT_FALSE(readme.empty()) << "README not readable at " << VISION_CORE_README_PATH;
    EXPECT_NE(readme.find(kMarkerStart), std::string::npos);
    EXPECT_NE(readme.find(kMarkerEnd), std::string::npos);
}

TEST(ReadmeModelTypesContract, EveryQuotedAliasRoutesViaTaskFactory) {
    const std::string readme = readReadme();
    ASSERT_FALSE(readme.empty());

    const std::string block = extractBlock(readme);
    ASSERT_FALSE(block.empty()) << "No content between TASKFACTORY_MODEL_LIST markers";

    const auto aliases = extractQuotedAliases(block);
    ASSERT_GE(aliases.size(), 20U) << "Implausibly few aliases extracted — marker or regex drift";

    const ModelInfo info = makeValidModelInfo();
    const TaskConfig config{};

    for (const auto& alias : aliases) {
        EXPECT_NO_THROW({
            auto task = TaskFactory::createTaskInstance(alias, info, config);
            EXPECT_NE(task, nullptr) << "Factory returned null for alias: " << alias;
        }) << "Alias documented in README but not routable: "
           << alias;
    }
}
