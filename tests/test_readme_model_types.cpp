#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/task_config.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#ifndef NEURIPLO_TASKS_README_PATH
#error "NEURIPLO_TASKS_README_PATH must be defined via target_compile_definitions"
#endif
#ifndef NEURIPLO_TASKS_TASK_FACTORY_PATH
#error "NEURIPLO_TASKS_TASK_FACTORY_PATH must be defined via target_compile_definitions"
#endif

using namespace neuriplo_tasks;

namespace {

constexpr const char* kMarkerStart = "<!-- TASKFACTORY_MODEL_LIST:START -->";
constexpr const char* kMarkerEnd = "<!-- TASKFACTORY_MODEL_LIST:END -->";

std::string readFile(const char* path) {
    std::ifstream in(path);
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

// Mirror of TaskFactory::normalizeModelType: drop whitespace, '-' and '_', lowercase.
std::string normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_') {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

std::set<std::string> extractQuotedAliases(const std::string& block) {
    static const std::regex kAlias(R"RX("([a-z0-9_-]+)")RX");
    std::set<std::string> out;
    for (std::sregex_iterator it(block.begin(), block.end(), kAlias), e; it != e; ++it) {
        out.insert((*it)[1].str());
    }
    return out;
}

std::string join(const std::set<std::string>& items) {
    std::string out;
    for (const auto& item : items) {
        out += out.empty() ? item : (", " + item);
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
    const std::string readme = readFile(NEURIPLO_TASKS_README_PATH);
    ASSERT_FALSE(readme.empty()) << "README not readable at " << NEURIPLO_TASKS_README_PATH;
    EXPECT_NE(readme.find(kMarkerStart), std::string::npos);
    EXPECT_NE(readme.find(kMarkerEnd), std::string::npos);
}

// Forward direction: every alias documented in the README routes via TaskFactory.
TEST(ReadmeModelTypesContract, EveryQuotedAliasRoutesViaTaskFactory) {
    const std::string readme = readFile(NEURIPLO_TASKS_README_PATH);
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

// Reverse direction: every standalone model-key string literal in
// task_factory.cpp must be documented in the README block. Catches a model
// being added to the factory but never listed in the README.
//
// Routing fragments are excluded: "seg"/"pose"/"splat" are internal
// substring matchers that never route on their own, and "resnet"/"tensorflow"
// are prefix/substring matchers documented in README prose rather than as
// quoted table rows.
TEST(ReadmeModelTypesContract, EveryFactoryKeyIsDocumentedInReadme) {
    static const std::set<std::string> kRoutingFragments = {"seg", "pose", "splat",    "resnet",
                                                            "kpt", "det",  "keypoint", "tensorflow"};

    const std::string factory = readFile(NEURIPLO_TASKS_TASK_FACTORY_PATH);
    ASSERT_FALSE(factory.empty()) << "task_factory.cpp not readable at " << NEURIPLO_TASKS_TASK_FACTORY_PATH;

    const std::string block = extractBlock(readFile(NEURIPLO_TASKS_README_PATH));
    ASSERT_FALSE(block.empty());

    std::set<std::string> documented;
    for (const auto& alias : extractQuotedAliases(block)) {
        documented.insert(normalize(alias));
    }

    static const std::regex kLiteral(R"RX("([a-z0-9-]{3,})")RX");
    std::set<std::string> missing;
    for (std::sregex_iterator it(factory.begin(), factory.end(), kLiteral), e; it != e; ++it) {
        const std::string key = normalize((*it)[1].str());
        if (key.empty() || kRoutingFragments.count(key) != 0) {
            continue;
        }
        if (documented.count(key) == 0) {
            missing.insert(key);
        }
    }

    EXPECT_TRUE(missing.empty()) << "Factory model keys absent from README block: " << join(missing);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
